"""The slice of the CUDA driver API that the NGX CUDA backend needs.

NGX reads its inputs through CUDA texture objects and writes its output through a
surface object, both of which have to be backed by CUDA arrays. This module binds
just enough of the driver API (``nvcuda.dll`` / ``libcuda.so.1``) with ctypes to
allocate those arrays and blit device memory in and out of them, so nothing here
needs the CUDA toolkit to be installed.

Image sources are whatever carries a device pointer: any DLPack producer (Dr.Jit
arrays and tensors, torch, cupy), anything exposing ``__cuda_array_interface__``,
or an explicit ``(pointer, width, height, channels)`` tuple.
"""

import ctypes
import sys
from typing import NamedTuple

CUDA_SUCCESS = 0
CU_AD_FORMAT_FLOAT = 0x20
CU_RESOURCE_TYPE_ARRAY = 0
CU_TR_ADDRESS_MODE_CLAMP = 1
CU_TR_FILTER_MODE_POINT = 0
CU_TRSF_NORMALIZED_COORDINATES = 2
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_ARRAY = 3


class CudaError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# driver API
# ---------------------------------------------------------------------------

class _ArrayDescriptor(ctypes.Structure):
    _fields_ = [("Width", ctypes.c_size_t), ("Height", ctypes.c_size_t),
                ("Format", ctypes.c_int), ("NumChannels", ctypes.c_uint32)]


class _ResourceDescUnion(ctypes.Union):
    _fields_ = [("hArray", ctypes.c_void_p), ("reserved", ctypes.c_int * 32)]


class _ResourceDesc(ctypes.Structure):
    _fields_ = [("resType", ctypes.c_int), ("res", _ResourceDescUnion),
                ("flags", ctypes.c_uint32)]


class _TextureDesc(ctypes.Structure):
    _fields_ = [("addressMode", ctypes.c_int * 3), ("filterMode", ctypes.c_int),
                ("flags", ctypes.c_uint32), ("maxAnisotropy", ctypes.c_uint32),
                ("mipmapFilterMode", ctypes.c_int), ("mipmapLevelBias", ctypes.c_float),
                ("minMipmapLevelClamp", ctypes.c_float), ("maxMipmapLevelClamp", ctypes.c_float),
                ("borderColor", ctypes.c_float * 4), ("reserved", ctypes.c_int * 12)]


class _Memcpy2D(ctypes.Structure):
    _fields_ = [("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t),
                ("srcMemoryType", ctypes.c_int), ("srcHost", ctypes.c_void_p),
                ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
                ("srcPitch", ctypes.c_size_t),
                ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
                ("dstMemoryType", ctypes.c_int), ("dstHost", ctypes.c_void_p),
                ("dstDevice", ctypes.c_uint64), ("dstArray", ctypes.c_void_p),
                ("dstPitch", ctypes.c_size_t),
                ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t)]


class _Driver:
    _instance = None

    def __init__(self):
        name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        try:
            self.lib = ctypes.CDLL(name)
        except OSError as e:
            raise CudaError(f"could not load {name}: {e}") from e
        ptr, u64, i32, u32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint32

        # The driver exports an ABI-versioned name for everything that changed
        # shape since CUDA 3.2, and cuda.h redirects those; the suffix is part of
        # the symbol name, not a pattern to guess (cuCtxGetDevice_v2 also exists,
        # but it is not the entry point cuda.h maps cuCtxGetDevice to).
        self.Init = self._bind("cuInit", [u32])
        self.CtxGetCurrent = self._bind("cuCtxGetCurrent", [ctypes.POINTER(ptr)])
        self.CtxPushCurrent = self._bind("cuCtxPushCurrent_v2", [ptr])
        self.CtxPopCurrent = self._bind("cuCtxPopCurrent_v2", [ctypes.POINTER(ptr)])
        self.CtxGetDevice = self._bind("cuCtxGetDevice", [ctypes.POINTER(i32)])
        self.ArrayCreate = self._bind("cuArrayCreate_v2",
                                      [ctypes.POINTER(ptr), ctypes.POINTER(_ArrayDescriptor)])
        self.ArrayDestroy = self._bind("cuArrayDestroy", [ptr])
        self.TexObjectCreate = self._bind("cuTexObjectCreate",
                                          [ctypes.POINTER(u64), ctypes.POINTER(_ResourceDesc),
                                           ctypes.POINTER(_TextureDesc), ptr])
        self.TexObjectDestroy = self._bind("cuTexObjectDestroy", [u64])
        self.SurfObjectCreate = self._bind("cuSurfObjectCreate",
                                           [ctypes.POINTER(u64), ctypes.POINTER(_ResourceDesc)])
        self.SurfObjectDestroy = self._bind("cuSurfObjectDestroy", [u64])
        self.Memcpy2DAsync = self._bind("cuMemcpy2DAsync_v2", [ctypes.POINTER(_Memcpy2D), ptr])
        self.StreamSynchronize = self._bind("cuStreamSynchronize", [ptr])
        self.GetErrorName = self._bind("cuGetErrorName", [i32, ctypes.POINTER(ctypes.c_char_p)])
        self.Init(0)

    def _bind(self, name, argtypes):
        fn = getattr(self.lib, name, None)
        if fn is None:
            raise CudaError(f"the CUDA driver does not export {name}")
        fn.argtypes, fn.restype = argtypes, ctypes.c_int
        return fn

    @classmethod
    def get(cls) -> "_Driver":
        if cls._instance is None:
            cls._instance = _Driver()
        return cls._instance


def driver() -> _Driver:
    return _Driver.get()


def check(result: int, what: str):
    if result != CUDA_SUCCESS:
        name = ctypes.c_char_p()
        if driver().GetErrorName(result, ctypes.byref(name)) == CUDA_SUCCESS and name.value:
            raise CudaError(f"{what} failed: {name.value.decode()}")
        raise CudaError(f"{what} failed with CUresult {result}")


def current_context() -> int:
    context = ctypes.c_void_p()
    check(driver().CtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
    if not context.value:
        raise CudaError("no CUDA context is current on this thread")
    return context.value


def context_device(context: int) -> int:
    """The device ordinal a context belongs to."""
    with ContextGuard(context):
        ordinal = ctypes.c_int(0)
        check(driver().CtxGetDevice(ctypes.byref(ordinal)), "cuCtxGetDevice")
        return ordinal.value


class ContextGuard:
    """Makes a context current for the duration of a block.

    A renderer that does its own context bookkeeping should push and pop through
    that instead, and pass its guard to the denoiser.
    """

    def __init__(self, context: int):
        self.context = context

    def __enter__(self):
        check(driver().CtxPushCurrent(ctypes.c_void_p(self.context)), "cuCtxPushCurrent")
        return self

    def __exit__(self, *args):
        previous = ctypes.c_void_p()
        driver().CtxPopCurrent(ctypes.byref(previous))
        return False


# ---------------------------------------------------------------------------
# image sources
# ---------------------------------------------------------------------------

class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


_DL_CUDA = (2, 13)  # kDLCUDA, kDLCUDAManaged
_DL_FLOAT = 2       # kDLFloat

_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


class ImageSource(NamedTuple):
    """A 2D block of float32 device memory: `pitch` is the row stride in bytes.

    `owner` keeps whatever handed out the pointer alive for as long as the source
    is used. For DLPack that is the capsule, whose destructor releases the
    producer's reference to the buffer.
    """
    pointer: int
    width: int
    height: int
    channels: int
    pitch: int
    owner: object = None


def _from_dlpack(value, name: str) -> ImageSource:
    capsule = value.__dlpack__()
    address = _capsule_pointer(capsule, b"dltensor")
    if not address:
        raise CudaError(f"{name}: __dlpack__() did not produce a 'dltensor' capsule")
    dl = ctypes.cast(address, ctypes.POINTER(_DLTensor))[0]
    if dl.device.device_type not in _DL_CUDA:
        raise CudaError(f"{name}: the data is not in CUDA memory "
                        f"(DLPack device type {dl.device.device_type})")
    if (dl.dtype.code, dl.dtype.bits, dl.dtype.lanes) != (_DL_FLOAT, 32, 1):
        raise CudaError(f"{name}: DLSS reads float32 images, got DLPack dtype "
                        f"({dl.dtype.code}, {dl.dtype.bits}, {dl.dtype.lanes})")
    # A flat buffer is read as a single row; reshape() gives it its real layout.
    if dl.ndim == 1:
        height, width, channels = 1, dl.shape[0], 1
    elif dl.ndim == 2:
        height, width, channels = dl.shape[0], dl.shape[1], 1
    elif dl.ndim == 3:
        height, width, channels = dl.shape[0], dl.shape[1], dl.shape[2]
    else:
        raise CudaError(f"{name}: expected a (height, width[, channels]) array, "
                        f"got {dl.ndim} dimensions")
    pitch = width * channels * 4
    if dl.strides:
        strides = [dl.strides[i] for i in range(dl.ndim)]
        if strides[-1] != 1 or (dl.ndim == 3 and strides[1] != channels):
            raise CudaError(f"{name}: rows must be tightly packed, got strides {strides}")
        if dl.ndim > 1:
            pitch = strides[0] * 4
    return ImageSource(dl.data + dl.byte_offset, width, height, channels, pitch, capsule)


def _from_cuda_array_interface(interface: dict, name: str) -> ImageSource:
    if interface["typestr"] not in ("<f4", "f4"):
        raise CudaError(f"{name}: DLSS reads float32 images, got {interface['typestr']}")
    shape = interface["shape"]
    if len(shape) == 1:
        height, width, channels = 1, shape[0], 1
    elif len(shape) == 2:
        height, width, channels = shape[0], shape[1], 1
    elif len(shape) == 3:
        height, width, channels = shape
    else:
        raise CudaError(f"{name}: expected a (height, width[, channels]) array, got {shape}")
    strides = interface.get("strides")
    if strides and strides[-1] != 4:
        raise CudaError(f"{name}: the innermost axis must be tightly packed")
    pitch = strides[0] if strides and len(shape) > 1 else width * channels * 4
    return ImageSource(interface["data"][0], width, height, channels, pitch)


def image_source(value, name: str = "image") -> ImageSource:
    """Resolve whatever the caller passed into a device pointer and a layout."""
    if isinstance(value, ImageSource):
        return value
    if isinstance(value, tuple):
        if len(value) != 4:
            raise CudaError(f"{name}: expected (pointer, width, height, channels)")
        pointer, width, height, channels = value
        return ImageSource(pointer, width, height, channels, width * channels * 4)
    if hasattr(value, "__dlpack__"):
        return _from_dlpack(value, name)
    if (interface := getattr(value, "__cuda_array_interface__", None)) is not None:
        return _from_cuda_array_interface(interface, name)
    raise CudaError(f"{name}: expected a DLPack or CUDA array interface object, or a "
                    f"(pointer, width, height, channels) tuple, got {type(value).__name__}")


def reshape(value, width: int, height: int, channels: int, name: str = "image") -> ImageSource:
    """Read a tightly packed buffer as a `width` x `height` x `channels` image."""
    source = image_source(value, name)
    count = source.width * source.height * source.channels
    if count != width * height * channels:
        raise CudaError(f"{name}: expected {width * height * channels} floats, got {count}")
    return source._replace(width=width, height=height, channels=channels,
                           pitch=width * channels * 4)


# ---------------------------------------------------------------------------
# CUDA arrays
# ---------------------------------------------------------------------------

class Array:
    """A CUDA array with the texture and surface objects NGX binds it through.

    NGX takes a pointer to the 64-bit object handle rather than the handle
    itself, which is what :meth:`texture_ptr` and :meth:`surface_ptr` hand out.
    """

    def __init__(self, width: int, height: int, channels: int):
        if channels not in (1, 2, 4):
            raise CudaError(f"CUDA arrays hold 1, 2 or 4 channels per element, not {channels}")
        self.width, self.height, self.channels = width, height, channels
        self.array = ctypes.c_void_p()
        self._texture = ctypes.c_uint64(0)
        self._surface = ctypes.c_uint64(0)

        descriptor = _ArrayDescriptor(Width=width, Height=height, Format=CU_AD_FORMAT_FLOAT,
                                      NumChannels=channels)
        check(driver().ArrayCreate(ctypes.byref(self.array), ctypes.byref(descriptor)),
              "cuArrayCreate")

        resource = _ResourceDesc()
        resource.resType = CU_RESOURCE_TYPE_ARRAY
        resource.res.hArray = self.array
        texture = _TextureDesc()
        texture.addressMode[:] = [CU_TR_ADDRESS_MODE_CLAMP] * 3
        texture.filterMode = CU_TR_FILTER_MODE_POINT
        texture.flags = CU_TRSF_NORMALIZED_COORDINATES
        check(driver().TexObjectCreate(ctypes.byref(self._texture), ctypes.byref(resource),
                                       ctypes.byref(texture), None), "cuTexObjectCreate")
        check(driver().SurfObjectCreate(ctypes.byref(self._surface), ctypes.byref(resource)),
              "cuSurfObjectCreate")

    def texture_ptr(self):
        return ctypes.byref(self._texture)

    def surface_ptr(self):
        return ctypes.byref(self._surface)

    def _copy(self, source: ImageSource, stream: int, to_array: bool):
        if source.width != self.width or source.height != self.height:
            raise CudaError(f"expected a {self.width}x{self.height} image, "
                            f"got {source.width}x{source.height}")
        if source.channels != self.channels:
            raise CudaError(f"expected {self.channels} channels, got {source.channels}")
        row = self.width * self.channels * 4
        op = _Memcpy2D(WidthInBytes=row, Height=self.height)
        if to_array:
            op.srcMemoryType = CU_MEMORYTYPE_DEVICE
            op.srcDevice, op.srcPitch = source.pointer, source.pitch
            op.dstMemoryType, op.dstArray = CU_MEMORYTYPE_ARRAY, self.array
        else:
            op.srcMemoryType, op.srcArray = CU_MEMORYTYPE_ARRAY, self.array
            op.dstMemoryType = CU_MEMORYTYPE_DEVICE
            op.dstDevice, op.dstPitch = source.pointer, source.pitch
        check(driver().Memcpy2DAsync(ctypes.byref(op), ctypes.c_void_p(stream)),
              "cuMemcpy2DAsync")

    def upload(self, source: ImageSource, stream: int = 0):
        self._copy(source, stream, to_array=True)

    def download(self, destination: ImageSource, stream: int = 0):
        self._copy(destination, stream, to_array=False)

    def destroy(self):
        if not self.array:
            return
        driver().SurfObjectDestroy(self._surface)
        driver().TexObjectDestroy(self._texture)
        driver().ArrayDestroy(self.array)
        self.array = ctypes.c_void_p()

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

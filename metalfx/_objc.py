"""The Metal objects the MetalFX effects are built out of, through PyObjC.

MetalFX works on ``MTLTexture`` objects, so this module owns the small amount of
Metal needed to make them, fill them and read them back, and nothing else: a
device, a pixel format table, and a texture wrapper that accepts whatever the
caller has (a texture it already owns, or any CPU array).

Textures are allocated with shared storage, which on the Apple silicon MetalFX
requires means the CPU and GPU address the same memory -- an upload is a memcpy,
not a transfer over a bus.
"""

import numpy as np

try:
    import Metal
    import MetalFX  # noqa: F401  (imported for its side effect of loading the framework)
    import objc
    from objc import simd
except ImportError as e:  # pragma: no cover - the package only loads on macOS
    raise ImportError("denoiser.metalfx needs macOS with pyobjc-framework-Metal and "
                      "pyobjc-framework-MetalFX installed") from e


class MetalFXError(RuntimeError):
    pass


# name -> (MTLPixelFormat, channels, dtype). The names are the ones the effects
# use for their inputs; add to this table to bind a format it does not cover.
FORMATS = {
    "r16_float":    (Metal.MTLPixelFormatR16Float,    1, np.float16),
    "rg16_float":   (Metal.MTLPixelFormatRG16Float,   2, np.float16),
    "rgba16_float": (Metal.MTLPixelFormatRGBA16Float, 4, np.float16),
    "r32_float":    (Metal.MTLPixelFormatR32Float,    1, np.float32),
    "rg32_float":   (Metal.MTLPixelFormatRG32Float,   2, np.float32),
    "rgba32_float": (Metal.MTLPixelFormatRGBA32Float, 4, np.float32),
    "rgba8_unorm":  (Metal.MTLPixelFormatRGBA8Unorm,  4, np.uint8),
    "bgra8_unorm":  (Metal.MTLPixelFormatBGRA8Unorm,  4, np.uint8),
}

USAGE = (Metal.MTLTextureUsageShaderRead | Metal.MTLTextureUsageShaderWrite
         | Metal.MTLTextureUsageRenderTarget)


def default_device():
    """The system default MTLDevice."""
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise MetalFXError("no Metal device available")
    return device


def as_object(handle):
    """A PyObjC proxy for `handle`, which may already be one or be an address.

    An address is what a rendering framework hands out for its native objects
    (slangpy's ``native_handle``, for instance), so this is the seam through
    which an externally owned MTLDevice or MTLTexture is adopted, no copy
    involved.
    """
    if handle is None or isinstance(handle, objc.objc_object):
        return handle
    if isinstance(handle, int):
        return objc.objc_object(c_void_p=handle)
    if hasattr(handle, "value") and isinstance(handle.value, int):  # ctypes pointer
        return objc.objc_object(c_void_p=handle.value)
    return handle  # anything else is assumed to already speak Objective-C


def to_simd(matrix) -> "simd.simd_float4x4":
    """A simd_float4x4 from any row-major 4x4 matrix (numpy, nested sequence)."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape != (4, 4):
        raise MetalFXError(f"expected a 4x4 matrix, got shape {m.shape}")
    # simd matrices are column-major: simd_float4x4 takes its columns in order.
    return simd.simd_float4x4(tuple(tuple(float(v) for v in m[:, c]) for c in range(4)))


def camel(name: str) -> str:
    """``diffuse_albedo`` -> ``DiffuseAlbedo``, the way MTLFX spells its setters."""
    return "".join(part.capitalize() for part in name.split("_"))


def to_numpy(value, name: str) -> np.ndarray:
    """Whatever the caller passed, as a numpy array, without leaving the CPU.

    Accepts numpy arrays, DLPack producers (Dr.Jit, torch) and anything with a
    ``numpy()`` method or the array protocol. GPU tensors have to be brought
    across first: there is no public way to reach the MTLBuffer behind a torch
    MPS tensor, so ``.cpu()`` is the caller's job.
    """
    if isinstance(value, np.ndarray):
        return value
    for attempt in (lambda: np.from_dlpack(value),
                    lambda: value.numpy(),
                    lambda: np.asarray(value)):
        try:
            return attempt()
        except Exception:
            continue
    raise MetalFXError(f"{name}: cannot read {type(value).__name__} as a CPU array; "
                       f"for a GPU tensor, move it to the host first")


class Texture:
    """An MTLTexture, either adopted from the caller or allocated here."""

    def __init__(self, device, width: int, height: int, format: str,
                 texture=None, label: str | None = None):
        if format not in FORMATS:
            raise MetalFXError(f"unknown pixel format {format!r}, extend FORMATS")
        self.format = format
        self.pixel_format, self.channels, self.dtype = FORMATS[format]
        self.width, self.height = width, height
        self.owned = texture is None

        if not self.owned:
            self.texture = as_object(texture)
            return

        descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            self.pixel_format, width, height, False)
        descriptor.setUsage_(USAGE)
        descriptor.setStorageMode_(Metal.MTLStorageModeShared)
        self.texture = device.newTextureWithDescriptor_(descriptor)
        if self.texture is None:
            raise MetalFXError(f"could not allocate a {width}x{height} {format} texture")
        if label:
            self.texture.setLabel_(label)

    @property
    def shape(self):
        return (self.height, self.width, self.channels)

    @property
    def row_bytes(self) -> int:
        return self.width * self.channels * np.dtype(self.dtype).itemsize

    def write(self, source, name: str = "image"):
        """Fill the texture from any CPU array shaped (height, width[, channels])."""
        array = to_numpy(source, name)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.shape[:2] != (self.height, self.width):
            raise MetalFXError(f"{name}: expected {self.height}x{self.width}, "
                               f"got {array.shape[0]}x{array.shape[1]}")
        if array.shape[2] != self.channels:
            raise MetalFXError(f"{name}: expected {self.channels} channel(s), "
                               f"got {array.shape[2]}")
        array = np.ascontiguousarray(array, dtype=self.dtype)
        region = Metal.MTLRegionMake2D(0, 0, self.width, self.height)
        # A memoryview hands Metal the array's own storage; `tobytes()` here would
        # copy the whole image once more on the way in.
        self.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region, 0, memoryview(array.reshape(-1)), self.row_bytes)

    def read(self, out: np.ndarray | None = None) -> np.ndarray:
        """The texture contents as a (height, width, channels) numpy array.

        `out` is written in place when given, which is what keeps a steady-state
        loop from allocating an image per frame.
        """
        if out is None:
            out = np.empty(self.shape, dtype=self.dtype)
        elif out.shape != self.shape or out.dtype != np.dtype(self.dtype):
            raise MetalFXError(f"expected an {self.shape} {np.dtype(self.dtype).name} "
                               f"array to read into, got {out.shape} {out.dtype}")
        elif not out.flags.c_contiguous:
            raise MetalFXError("the array to read into must be C-contiguous")
        region = Metal.MTLRegionMake2D(0, 0, self.width, self.height)
        self.texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
            memoryview(out.reshape(-1)), self.row_bytes, region, 0)
        return out

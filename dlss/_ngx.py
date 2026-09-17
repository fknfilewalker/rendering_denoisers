"""The NGX API of the NVIDIA driver, bound with ctypes.

The NGX component of the driver (``_nvngx.dll`` on Windows,
``libnvidia-ngx.so.1`` on Linux) exports every backend of the NGX API as flat C
entry points, so no C++ glue is needed: ctypes binds the driver directly. This
module holds what the CUDA and Vulkan backends share -- the loader, the parameter
object, feature discovery -- and :mod:`rendering_denoisers.dlss.cuda` and
:mod:`rendering_denoisers.dlss.vulkan` build the denoisers on top.

Two details deviate from the DLSS SDK headers:

* The driver-level Init entry points take ``const NVSDK_NGX_FeatureCommonInfo *``
  where the SDK's static library exposes ``const NVSDK_NGX_Parameter *``, and
  Shutdown1 gained an ``unsigned int &`` out parameter. This mirrors the
  PFN_NVSDK_NGX_CORE_* typedefs in the SDK's nvsdk_ngx_standalone_cuda.h, and
  holds for the Vulkan entry points too: the driver reads that argument at
  +0x00/+0x08 and, gated on SDK version >= 0x13 and >= 0x14, at +0x10 and
  +0x18..+0x24, which is exactly the NVSDK_NGX_FeatureCommonInfo layout.
* NVSDK_NGX_Parameter is a pure-virtual C++ class, and the flat
  NVSDK_NGX_Parameter_Set* wrappers live in the SDK's static library rather than
  in the driver. :class:`Parameters` dispatches through the vtable instead, with
  the slot order picked per platform (MSVC emits overload groups in reverse
  declaration order, the Itanium ABI in declaration order) and validated by a
  round trip before anything else is called.
"""

import ctypes
import os
import pathlib
import platform
import sys
import tempfile

VERSION_API = 0x0000015
FEATURE_RAY_RECONSTRUCTION = 13

RESULT_SUCCESS = 0x1
_RESULT_FAIL = 0xBAD00000
RESULT_NAMES = {
    RESULT_SUCCESS: "Success",
    _RESULT_FAIL: "Fail",
    _RESULT_FAIL | 1: "FeatureNotSupported",
    _RESULT_FAIL | 2: "PlatformError",
    _RESULT_FAIL | 3: "FeatureAlreadyExists",
    _RESULT_FAIL | 4: "FeatureNotFound",
    _RESULT_FAIL | 5: "InvalidParameter",
    _RESULT_FAIL | 6: "ScratchBufferTooSmall",
    _RESULT_FAIL | 7: "NotInitialized",
    _RESULT_FAIL | 8: "UnsupportedInputFormat",
    _RESULT_FAIL | 9: "RWFlagMissing",
    _RESULT_FAIL | 10: "MissingInput",
    _RESULT_FAIL | 11: "UnableToInitializeFeature",
    _RESULT_FAIL | 12: "OutOfDate",
    _RESULT_FAIL | 13: "OutOfGPUMemory",
    _RESULT_FAIL | 14: "UnsupportedFormat",
    _RESULT_FAIL | 15: "UnableToWriteToAppDataPath",
    _RESULT_FAIL | 16: "UnsupportedParameter",
    _RESULT_FAIL | 17: "Denied",
    _RESULT_FAIL | 18: "NotImplemented",
}

# NVSDK_NGX_Feature_Support_Result, a bitfield (0 means supported)
SUPPORT_NAMES = {
    1: "check not present",
    2: "driver version unsupported",
    4: "adapter unsupported",
    8: "OS version below minimum",
    16: "not implemented",
}


class DLSSError(RuntimeError):
    pass


def result_name(result: int) -> str:
    code = result & 0xFFFFFFFF
    return f"{RESULT_NAMES.get(code, 'unknown')} (0x{code:08x})"


def check(result: int, what: str):
    if result != RESULT_SUCCESS:
        raise DLSSError(f"{what} failed: {result_name(result)}")


# ---------------------------------------------------------------------------
# ctypes declarations
# ---------------------------------------------------------------------------

class PathListInfo(ctypes.Structure):
    _fields_ = [("Path", ctypes.POINTER(ctypes.c_wchar_p)), ("Length", ctypes.c_uint32)]


LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_int)


class LoggingInfo(ctypes.Structure):
    _fields_ = [("LoggingCallback", LOG_CALLBACK), ("MinimumLoggingLevel", ctypes.c_int),
                ("DisableOtherLoggingSinks", ctypes.c_bool)]


class FeatureCommonInfo(ctypes.Structure):
    _fields_ = [("PathListInfo", PathListInfo), ("InternalData", ctypes.c_void_p),
                ("LoggingInfo", LoggingInfo)]


class ProjectIdDescription(ctypes.Structure):
    _fields_ = [("ProjectId", ctypes.c_char_p), ("EngineType", ctypes.c_int),
                ("EngineVersion", ctypes.c_char_p)]


class ApplicationIdentifierValue(ctypes.Union):
    _fields_ = [("ProjectDesc", ProjectIdDescription), ("ApplicationId", ctypes.c_uint64)]


class ApplicationIdentifier(ctypes.Structure):
    _fields_ = [("IdentifierType", ctypes.c_int), ("v", ApplicationIdentifierValue)]


class FeatureDiscoveryInfo(ctypes.Structure):
    _fields_ = [("SDKVersion", ctypes.c_int), ("FeatureID", ctypes.c_int),
                ("Identifier", ApplicationIdentifier), ("ApplicationDataPath", ctypes.c_wchar_p),
                ("FeatureInfo", ctypes.POINTER(FeatureCommonInfo))]


class FeatureRequirement(ctypes.Structure):
    _fields_ = [("FeatureSupported", ctypes.c_int), ("MinHWArchitecture", ctypes.c_uint32),
                ("MinOSVersion", ctypes.c_char * 255)]


class VkExtensionProperties(ctypes.Structure):
    _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", ctypes.c_uint32)]


class CUDADevice(ctypes.Structure):
    """NVSDK_NGX_CUDADevice: the context and stream NGX should work on."""
    _fields_ = [("cudaContext", ctypes.c_void_p), ("cudaStream", ctypes.c_void_p)]


# ---------------------------------------------------------------------------
# driver library
# ---------------------------------------------------------------------------

def driver_library_path() -> str | None:
    """Where the NGX component of the display driver lives, if it is installed."""
    if sys.platform == "win32":
        import winreg
        for key in (r"System\CurrentControlSet\Services\nvlddmkm\Parameters\NGXCore",
                    r"System\CurrentControlSet\Services\nvlddmkm\NGXCore"):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                    directory = pathlib.Path(winreg.QueryValueEx(handle, "NGXPath")[0])
            except OSError:
                continue
            arm = platform.machine().lower() in ("arm64", "aarch64")
            path = directory / ("_arm64_nvngx.dll" if arm else "_nvngx.dll")
            if path.is_file():
                return str(path)
        return None
    return "libnvidia-ngx.so.1"


class Driver:
    """The NGX entry points of one backend, as exported by the display driver."""

    _instances: dict[str, "Driver"] = {}

    def __init__(self, backend: str, path: str):
        self.backend = backend
        self.path = path
        self.lib = ctypes.CDLL(path)
        ptr = ctypes.c_void_p

        self.AllocateParameters = self._bind("AllocateParameters", [ctypes.POINTER(ptr)])
        self.DestroyParameters = self._bind("DestroyParameters", [ptr])
        self.ReleaseFeature = self._bind("ReleaseFeature", [ptr])
        (self._bind_vulkan if backend == "VULKAN" else self._bind_cuda)()

    def _bind(self, name, argtypes, restype=ctypes.c_uint32):
        fn = getattr(self.lib, f"NVSDK_NGX_{self.backend}_{name}", None)
        if fn is None:
            raise DLSSError(f"{self.path} does not export NVSDK_NGX_{self.backend}_{name}")
        fn.argtypes, fn.restype = argtypes, restype
        return fn

    def _bind_cuda(self):
        ptr, u32, u64, i32 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int
        device = ctypes.POINTER(CUDADevice)
        # Init_Ext1 and Shutdown1 use the driver-level signatures, see module docstring.
        self.Init = self._bind("Init_Ext1", [u64, ctypes.c_wchar_p, device, i32,
                                             ctypes.POINTER(FeatureCommonInfo)])
        self.Shutdown = self._bind("Shutdown1", [device, ctypes.POINTER(u32)])
        self.CreateFeature = self._bind("CreateFeature1", [device, i32, ptr, ctypes.POINTER(ptr)])
        self.EvaluateFeature = self._bind("EvaluateFeature", [ptr, ptr, ptr])
        self.GetFeatureRequirements = self._bind(
            "GetFeatureRequirements",
            [i32, ctypes.POINTER(FeatureDiscoveryInfo), ctypes.POINTER(FeatureRequirement)])

    def _bind_vulkan(self):
        ptr, u32, u64, i32 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int
        self.Init = self._bind("Init_Ext2", [u64, ctypes.c_wchar_p, ptr, ptr, ptr, ptr, ptr, i32,
                                             ctypes.POINTER(FeatureCommonInfo)])
        self.Shutdown = self._bind("Shutdown1", [ptr, ctypes.POINTER(u32)])
        self.CreateFeature = self._bind("CreateFeature1", [ptr, ptr, i32, ptr, ctypes.POINTER(ptr)])
        self.EvaluateFeature = self._bind("EvaluateFeature", [ptr, ptr, ptr, ptr])
        self.GetFeatureRequirements = self._bind(
            "GetFeatureRequirements",
            [ptr, ptr, ctypes.POINTER(FeatureDiscoveryInfo), ctypes.POINTER(FeatureRequirement)])
        self.GetFeatureInstanceExtensionRequirements = self._bind(
            "GetFeatureInstanceExtensionRequirements",
            [ctypes.POINTER(FeatureDiscoveryInfo), ctypes.POINTER(u32),
             ctypes.POINTER(ctypes.POINTER(VkExtensionProperties))])
        self.GetFeatureDeviceExtensionRequirements = self._bind(
            "GetFeatureDeviceExtensionRequirements",
            [ptr, ptr, ctypes.POINTER(FeatureDiscoveryInfo), ctypes.POINTER(u32),
             ctypes.POINTER(ctypes.POINTER(VkExtensionProperties))])
        self.RequiredExtensions = self._bind(
            "RequiredExtensions",
            [ctypes.POINTER(u32), ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
             ctypes.POINTER(u32), ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p))])

    @classmethod
    def get(cls, backend: str = "CUDA") -> "Driver":
        if backend not in cls._instances:
            path = driver_library_path()
            if path is None:
                raise DLSSError("no NVIDIA NGX driver component found")
            try:
                cls._instances[backend] = Driver(backend, path)
            except OSError as e:
                raise DLSSError(f"could not load {path}: {e}") from e
        return cls._instances[backend]


def driver_available(backend: str = "CUDA") -> bool:
    """Whether the NGX component of the NVIDIA driver can be loaded at all."""
    try:
        Driver.get(backend)
        return True
    except DLSSError:
        return False


# ---------------------------------------------------------------------------
# NVSDK_NGX_Parameter
# ---------------------------------------------------------------------------

# Declaration order in nvsdk_ngx_params.h: 8 Set overloads (ull, float, double,
# uint, int, ID3D11Resource*, ID3D12Resource*, void*), the 8 matching Get
# overloads, then Reset.
_SLOTS_DECL = {"ull": 0, "float": 1, "double": 2, "uint": 3, "int": 4, "ptr": 7,
               "get_ull": 8, "get_float": 9, "get_double": 10, "get_uint": 11,
               "get_int": 12, "get_ptr": 15, "reset": 16}
# MSVC emits each overload group in reverse declaration order.
_SLOTS_MSVC = {"ull": 7, "float": 6, "double": 5, "uint": 4, "int": 3, "ptr": 0,
               "get_ull": 15, "get_float": 14, "get_double": 13, "get_uint": 12,
               "get_int": 11, "get_ptr": 8, "reset": 16}


class Parameters:
    """Calls an NVSDK_NGX_Parameter through its vtable."""

    _slots = None

    def __init__(self, ptr: int):
        self.ptr = ptr
        self._vtable = ctypes.cast(ctypes.c_void_p(ptr),
                                   ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        if Parameters._slots is None:
            Parameters._slots = self._resolve_slots()

    def _call(self, slot: int, argtype, name: bytes, value, restype=None):
        fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_char_p, argtype)(self._vtable[slot])
        return fn(ctypes.c_void_p(self.ptr), name, value)

    def _resolve_slots(self) -> dict:
        """Pick the vtable layout for this platform and prove it round-trips.

        Only slots 0..7 are ever called with a value, and those are Set overloads
        under both candidate layouts, so they never dereference the value. The
        value passed is a real address anyway, and the Get probes only write into
        a local, so a wrong guess fails the check instead of corrupting memory.
        """
        slots = _SLOTS_MSVC if sys.platform == "win32" else _SLOTS_DECL
        probe = b"rtm.abi.probe"
        scratch = ctypes.c_double(0.0)

        self._call(slots["ptr"], ctypes.c_void_p, probe, ctypes.byref(scratch))
        out = ctypes.c_void_p(0)
        rv = self._call(slots["get_ptr"], ctypes.POINTER(ctypes.c_void_p), probe,
                        ctypes.byref(out), restype=ctypes.c_uint32)
        if rv != RESULT_SUCCESS or out.value != ctypes.addressof(scratch):
            raise DLSSError("unexpected NVSDK_NGX_Parameter vtable layout (pointer round trip)")

        self._call(slots["float"], ctypes.c_float, probe, 0.5)
        wide = ctypes.c_double(0.0)  # oversized, in case a Get overload writes 8 bytes
        as_float = ctypes.cast(ctypes.byref(wide), ctypes.POINTER(ctypes.c_float))
        rv = self._call(slots["get_float"], ctypes.POINTER(ctypes.c_float), probe,
                        as_float, restype=ctypes.c_uint32)
        if rv != RESULT_SUCCESS or as_float[0] != 0.5:
            raise DLSSError("unexpected NVSDK_NGX_Parameter vtable layout (float round trip)")
        return slots

    def set_int(self, name: str, value: int):
        self._call(self._slots["int"], ctypes.c_int, name.encode(), value)

    def set_uint(self, name: str, value: int):
        self._call(self._slots["uint"], ctypes.c_uint32, name.encode(), value)

    def set_float(self, name: str, value: float):
        self._call(self._slots["float"], ctypes.c_float, name.encode(), value)

    def set_ptr(self, name: str, value):
        self._call(self._slots["ptr"], ctypes.c_void_p, name.encode(),
                   ctypes.cast(value, ctypes.c_void_p) if value is not None else None)

    def get_uint(self, name: str) -> int | None:
        out = ctypes.c_uint32(0)
        rv = self._call(self._slots["get_uint"], ctypes.POINTER(ctypes.c_uint32), name.encode(),
                        ctypes.byref(out), restype=ctypes.c_uint32)
        return out.value if rv == RESULT_SUCCESS else None


# ---------------------------------------------------------------------------
# application identity and feature discovery
# ---------------------------------------------------------------------------

def _log(message: bytes, level: int, feature: int):
    print(f"[dlss] {message.decode(errors='replace').rstrip()}")


class Common:
    """NVSDK_NGX_FeatureCommonInfo plus the buffers it points at.

    `library_path` is where NGX should look for the Ray Reconstruction snippet.
    NGX would otherwise search next to the running executable, which under
    ctypes it reports as Python's own libffi.
    """

    def __init__(self, library_path: str | os.PathLike | None, verbose: bool):
        paths = []
        if library_path is not None:
            paths.append(str(pathlib.Path(library_path).absolute()))
        elif (env := os.environ.get("MI_DLSS_LIBRARY_PATH")) is not None:
            paths.extend(str(pathlib.Path(p).absolute()) for p in env.split(os.pathsep) if p)
        self._paths = (ctypes.c_wchar_p * max(1, len(paths)))(*paths)
        self._callback = LOG_CALLBACK(_log)
        self.info = FeatureCommonInfo()
        self.info.PathListInfo.Path = self._paths
        self.info.PathListInfo.Length = len(paths)
        self.info.LoggingInfo.LoggingCallback = self._callback
        self.info.LoggingInfo.MinimumLoggingLevel = 1 if verbose else 0
        self.info.LoggingInfo.DisableOtherLoggingSinks = False

    def discovery(self, application_id: int, data_path: str) -> FeatureDiscoveryInfo:
        info = FeatureDiscoveryInfo()
        info.SDKVersion = VERSION_API
        info.FeatureID = FEATURE_RAY_RECONSTRUCTION
        info.Identifier.IdentifierType = 0  # Application_Id
        info.Identifier.v.ApplicationId = application_id
        info.ApplicationDataPath = data_path
        info.FeatureInfo = ctypes.pointer(self.info)
        return info


def application_id(value: int | None = None) -> int:
    if value is not None:
        return value
    if (env := os.environ.get("MI_DLSS_APPLICATION_ID")) is not None:
        return int(env, 0)
    return 0x4D495453  # 'MITS', the id Mitsuba's own DLSS denoiser reports


def data_path(value: str | os.PathLike | None = None) -> str:
    path = pathlib.Path(value or os.environ.get("MI_DLSS_APP_DATA_PATH")
                        or pathlib.Path(tempfile.gettempdir()) / "rtm-dlss")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def support_reason(requirement: FeatureRequirement) -> str:
    reasons = [text for bit, text in SUPPORT_NAMES.items() if requirement.FeatureSupported & bit]
    return ", ".join(reasons) or f"0x{requirement.FeatureSupported:x}"

"""DLSS Ray Reconstruction over Vulkan images.

Where the CUDA backend stages every buffer through a CUDA array it owns, NGX
reads and writes Vulkan images in place: each buffer is a :class:`Resource`
naming a ``VkImage``, a ``VkImageView`` over it, its ``VkFormat`` and its extent,
and the evaluation is recorded into a ``VkCommandBuffer`` the caller provides.
Nothing is copied, so this is the cheaper backend when the renderer already owns
its images -- but it is also the one with the strings attached:

* The instance and device extensions NGX needs must be enabled when the instance
  and the device are created, which is *before* a denoiser can exist. Ask for
  them with :func:`required_extensions`, which needs no NGX initialisation.
* The caller owns the image layouts. Inputs must be in
  ``VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`` and the output in
  ``VK_IMAGE_LAYOUT_GENERAL`` when the recorded commands execute, and the output
  image needs ``VK_IMAGE_USAGE_STORAGE_BIT``.

Handles are plain integers, so this works with any Vulkan binding -- a framework
that hands out native handles, ``vulkan``/``pyvk``, or ctypes::

    from rendering_denoisers.dlss import vulkan

    ext = vulkan.required_extensions()           # before creating the device
    dlss = vulkan.DLSSDenoiser(instance, physical_device, device, (960, 540), (1920, 1080))
    dlss(command_buffer, color=color, output=target, depth=depth,
         motion_vectors=mvec, normals=normals, diffuse_albedo=diffuse,
         specular_albedo=specular, jitter=jitter, reset=camera_moved)

DLSS-D is *temporal*: it wants one sample per pixel with a known sub-pixel jitter,
screen-space motion vectors and separated guide buffers, not an accumulated image.
Reset the history whenever the camera or the scene jumps.
"""

import ctypes
from typing import NamedTuple

from . import _download
from ._feature import Feature, FEATURE_RAY_RECONSTRUCTION, QUALITY
from ._ngx import (Common, DLSSError, Driver, FeatureRequirement, Parameters, RESULT_SUCCESS,
                   VERSION_API, VkExtensionProperties, check, result_name, support_reason)
from ._ngx import application_id as _application_id, data_path as _data_path

__all__ = ["DLSSDenoiser", "DLSSError", "QUALITY", "Resource", "is_available",
           "required_extensions", "support_status", "driver_available"]

VK_IMAGE_ASPECT_COLOR_BIT = 0x1
_RESOURCE_TYPE_IMAGEVIEW = 0

# The VkFormats NGX is ever handed here, for callers that would rather name a
# format than look its number up. Anything else can be passed as a raw int.
VK_FORMAT = {
    "r8_unorm": 9, "rg8_unorm": 16, "rgba8_unorm": 37, "rgba8_srgb": 43,
    "bgra8_unorm": 44, "bgra8_srgb": 50, "r16_unorm": 70, "r16_float": 76,
    "rg16_unorm": 77, "rg16_float": 83, "rgba16_unorm": 91, "rgba16_float": 97,
    "r32_float": 100, "rg32_float": 103, "rgb32_float": 106, "rgba32_float": 109,
    "rgb10a2_unorm": 64, "r11g11b10_float": 122, "rgb9e5_ufloat": 123,
    "d16_unorm": 124, "d32_float": 126,
}


def driver_available() -> bool:
    """Whether the Vulkan NGX entry points of the driver can be loaded."""
    from ._ngx import driver_available as _available
    return _available("VULKAN")


class Resource(NamedTuple):
    """One Vulkan image NGX reads or writes.

    Args:
        image: The ``VkImage`` handle.
        view: A ``VkImageView`` over its first mip and layer.
        format: Its ``VkFormat``, as a number or a key of :data:`VK_FORMAT`.
        width, height: Its extent in pixels.
    """
    image: int
    view: int
    format: int | str
    width: int
    height: int


def _resource(value, name: str) -> Resource:
    if isinstance(value, Resource):
        return value
    if isinstance(value, dict):
        return Resource(**value)
    if isinstance(value, (tuple, list)) and len(value) == 5:
        return Resource(*value)
    raise DLSSError(f"{name}: expected a Resource, a dict or a "
                    f"(image, view, format, width, height) tuple, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# NGX resource descriptor
# ---------------------------------------------------------------------------

class _VkImageSubresourceRange(ctypes.Structure):
    _fields_ = [("aspectMask", ctypes.c_uint32), ("baseMipLevel", ctypes.c_uint32),
                ("levelCount", ctypes.c_uint32), ("baseArrayLayer", ctypes.c_uint32),
                ("layerCount", ctypes.c_uint32)]


class _ImageViewInfo(ctypes.Structure):
    _fields_ = [("ImageView", ctypes.c_void_p), ("Image", ctypes.c_void_p),
                ("SubresourceRange", _VkImageSubresourceRange), ("Format", ctypes.c_int),
                ("Width", ctypes.c_uint32), ("Height", ctypes.c_uint32)]


class _BufferInfo(ctypes.Structure):
    _fields_ = [("Buffer", ctypes.c_void_p), ("SizeInBytes", ctypes.c_uint32)]


class _ResourceUnion(ctypes.Union):
    _fields_ = [("ImageViewInfo", _ImageViewInfo), ("BufferInfo", _BufferInfo)]


class _Resource(ctypes.Structure):
    _fields_ = [("Resource", _ResourceUnion), ("Type", ctypes.c_int),
                ("ReadWrite", ctypes.c_bool)]


def _descriptor(resource: Resource, read_write: bool) -> _Resource:
    out = _Resource()
    info = out.Resource.ImageViewInfo
    info.ImageView = ctypes.c_void_p(resource.view)
    info.Image = ctypes.c_void_p(resource.image)
    info.SubresourceRange = _VkImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    info.Format = VK_FORMAT[resource.format] if isinstance(resource.format, str) \
        else resource.format
    info.Width, info.Height = resource.width, resource.height
    out.Type = _RESOURCE_TYPE_IMAGEVIEW
    out.ReadWrite = read_write
    return out


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _discovery(library_path, download, version, application_id, data_path):
    driver = Driver.get("VULKAN")
    path = _download.resolve(library_path, download=download, version=version)
    common = Common(path, verbose=False)
    return driver, common, common.discovery(_application_id(application_id),
                                            _data_path(data_path))


def support_status(instance: int, physical_device: int, *, library_path=None,
                   download: bool = True, version: str | None = None, application_id=None,
                   data_path=None) -> tuple[bool, str]:
    """Ask the driver whether Ray Reconstruction runs here, as (supported, reason).

    This needs no NGX initialisation, only a ``VkInstance`` and the
    ``VkPhysicalDevice`` the denoiser would run on.
    """
    try:
        driver, _common, info = _discovery(library_path, download, version, application_id,
                                           data_path)
        requirement = FeatureRequirement()
        rv = driver.GetFeatureRequirements(ctypes.c_void_p(instance),
                                           ctypes.c_void_p(physical_device),
                                           ctypes.byref(info), ctypes.byref(requirement))
        if rv != RESULT_SUCCESS:
            return False, f"GetFeatureRequirements: {result_name(rv)}"
        if requirement.FeatureSupported != 0:
            return False, support_reason(requirement)
        return True, "supported"
    except (DLSSError, _download.DownloadError) as e:
        return False, str(e)


def is_available(instance: int, physical_device: int, **kwargs) -> bool:
    """Whether DLSS Ray Reconstruction can be used on this physical device."""
    return support_status(instance, physical_device, **kwargs)[0]


def _feature_extensions(driver, info, instance, physical_device) -> dict | None:
    """Per-feature extension queries, or None if the driver does not implement them."""
    def query(call, *args):
        count = ctypes.c_uint32(0)
        properties = ctypes.POINTER(VkExtensionProperties)()
        rv = call(*args, ctypes.byref(info), ctypes.byref(count), ctypes.byref(properties))
        if rv != RESULT_SUCCESS:
            return None
        return [properties[i].extensionName.decode() for i in range(count.value)]

    if (instance_extensions := query(driver.GetFeatureInstanceExtensionRequirements)) is None:
        return None
    if instance is None or physical_device is None:
        return {"instance": instance_extensions, "device": []}
    device_extensions = query(driver.GetFeatureDeviceExtensionRequirements,
                              ctypes.c_void_p(instance), ctypes.c_void_p(physical_device))
    if device_extensions is None:
        return None
    return {"instance": instance_extensions, "device": device_extensions}


def required_extensions(instance: int | None = None, physical_device: int | None = None, *,
                        library_path=None, download: bool = True, version: str | None = None,
                        application_id=None, data_path=None) -> dict[str, list[str]]:
    """The Vulkan extensions Ray Reconstruction needs, as ``{"instance", "device"}``.

    Call this before creating the instance and the device, and enable what it
    returns: NGX cannot add extensions to a device that already exists.

    The per-feature queries are tried first; they need an instance and a physical
    device for the device extension list, and answer NotImplemented on drivers
    that do not have them, in which case this falls back to
    ``NVSDK_NGX_VULKAN_RequiredExtensions``, which needs neither and covers both
    lists.
    """
    driver, _common, info = _discovery(library_path, download, version, application_id,
                                       data_path)
    if (result := _feature_extensions(driver, info, instance, physical_device)) is not None:
        return result

    u32, names = ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p)
    instance_count, instance_names = u32(0), names()
    device_count, device_names = u32(0), names()
    check(driver.RequiredExtensions(ctypes.byref(instance_count), ctypes.byref(instance_names),
                                    ctypes.byref(device_count), ctypes.byref(device_names)),
          "NVSDK_NGX_VULKAN_RequiredExtensions")
    return {"instance": [instance_names[i].decode() for i in range(instance_count.value)],
            "device": [device_names[i].decode() for i in range(device_count.value)]}


# ---------------------------------------------------------------------------
# the denoiser
# ---------------------------------------------------------------------------

class DLSSDenoiser(Feature):
    """The NVIDIA DLSS Ray Reconstruction denoiser, on Vulkan images.

    Args:
        instance: The ``VkInstance``, created with the instance extensions of
            :func:`required_extensions`.
        physical_device: The ``VkPhysicalDevice`` to run on.
        device: The ``VkDevice``, created with the device extensions of
            :func:`required_extensions`.
        command_buffer: A ``VkCommandBuffer`` in the recording state, used once
            to create the feature. Must be submitted and completed before the
            first call.

    The feature configuration and the snippet options are those of
    :class:`rendering_denoisers.dlss.cuda.DLSSDenoiser`.
    """

    def __init__(self, instance: int, physical_device: int, device: int, command_buffer: int,
                 input_size, output_size=None, quality: str | int = "high", *,
                 library_path=None, download: bool = True, version: str | None = None,
                 application_id=None, data_path=None, verbose: bool = False, **feature):
        super().__init__(input_size, output_size, quality, **feature)
        self.instance, self.physical_device, self.device = instance, physical_device, device

        supported, reason = support_status(instance, physical_device, library_path=library_path,
                                           download=download, version=version,
                                           application_id=application_id, data_path=data_path)
        if not supported:
            raise DLSSError(
                f"DLSS Ray Reconstruction is not available on this system: {reason}. It needs an "
                f"RTX GPU, driver 590 or newer, and the Ray Reconstruction library "
                f"(nvngx_dlssd.dll / libnvidia-ngx-dlssd.so), which the MI_DLSS_LIBRARY_PATH "
                f"environment variable can point at")

        self.library_path = _download.resolve(library_path, download=download, version=version)
        self._common = Common(self.library_path, verbose)  # NGX keeps the pointer, keep it alive
        self._driver = Driver.get("VULKAN")
        self._feature = ctypes.c_void_p()

        check(self._driver.Init(_application_id(application_id), _data_path(data_path),
                                ctypes.c_void_p(instance), ctypes.c_void_p(physical_device),
                                ctypes.c_void_p(device), None, None, VERSION_API,
                                ctypes.byref(self._common.info)),
              "NVSDK_NGX_VULKAN_Init_Ext2")
        try:
            self._create_feature(command_buffer)
        except Exception:
            self._shutdown()
            raise

    # -- NGX lifecycle ------------------------------------------------------

    def _parameters(self) -> Parameters:
        out = ctypes.c_void_p()
        check(self._driver.AllocateParameters(ctypes.byref(out)), "AllocateParameters")
        return Parameters(out.value)

    def _create_feature(self, command_buffer: int):
        params = self._parameters()
        self.fill_create_params(params)
        rv = self._driver.CreateFeature(ctypes.c_void_p(self.device),
                                        ctypes.c_void_p(command_buffer),
                                        FEATURE_RAY_RECONSTRUCTION,
                                        ctypes.c_void_p(params.ptr), ctypes.byref(self._feature))
        self._driver.DestroyParameters(ctypes.c_void_p(params.ptr))
        check(rv, "NVSDK_NGX_VULKAN_CreateFeature1")

    def _shutdown(self):
        remaining = ctypes.c_uint32(0)
        self._driver.Shutdown(ctypes.c_void_p(self.device), ctypes.byref(remaining))

    def release(self):
        """Destroy the NGX feature. The device must be idle."""
        if not self._feature:
            return
        self._driver.ReleaseFeature(self._feature)
        self._feature = ctypes.c_void_p()
        self._shutdown()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # -- evaluation ---------------------------------------------------------

    def __call__(self, command_buffer: int, *, color, output, depth, motion_vectors, normals,
                 diffuse_albedo, specular_albedo, roughness=None, specular_motion_vectors=None,
                 jitter=(0.0, 0.0), mv_scale=(1.0, 1.0), reset: bool = False,
                 pre_exposure: float = 1.0, exposure_scale: float = 1.0,
                 extra: dict | None = None):
        """Record a Ray Reconstruction evaluation into `command_buffer`.

        Every buffer is a :class:`Resource`; the inputs are at ``input_size`` and
        `output` is at ``output_size``. Their layouts are the caller's
        responsibility, see the module docstring.

        `jitter` is the sub-pixel camera offset this frame was rendered with,
        `mv_scale` converts the motion vector image into pixels, and `reset`
        drops the temporal history (it is implied for the first call). Anything
        else DLSS-D accepts goes in `extra`, keyed by NGX parameter name.
        """
        if not self._feature:
            raise DLSSError("this denoiser has been released")

        inputs = self.input_names({
            "color": color, "depth": depth, "motion_vectors": motion_vectors,
            "normals": normals, "roughness": roughness, "diffuse_albedo": diffuse_albedo,
            "specular_albedo": specular_albedo,
            "specular_motion_vectors": specular_motion_vectors}, extra,
            lambda value, name, _channels: _resource(value, name))

        # The descriptors have to outlive the call NGX reads them in, so they are
        # kept in a list rather than built inline.
        descriptors = [(name, _descriptor(resource, False))
                       for name, resource in inputs.items()]
        descriptors.append(("Output", _descriptor(_resource(output, "output"), True)))

        params = self._parameters()
        for name, descriptor in descriptors:
            params.set_ptr(name, ctypes.byref(descriptor))
        self.fill_eval_params(params, jitter, mv_scale, reset or self._reset,
                              pre_exposure, exposure_scale)
        rv = self._driver.EvaluateFeature(ctypes.c_void_p(command_buffer), self._feature,
                                          ctypes.c_void_p(params.ptr), None)
        self._driver.DestroyParameters(ctypes.c_void_p(params.ptr))
        check(rv, "NVSDK_NGX_VULKAN_EvaluateFeature")
        self._reset = False
        return output

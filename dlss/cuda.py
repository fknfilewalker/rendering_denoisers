"""DLSS Ray Reconstruction over CUDA device memory.

Every buffer is an *image source* -- anything exposing ``__dlpack__`` or
``__cuda_array_interface__`` (Dr.Jit, torch, cupy), or an explicit
``(pointer, width, height, channels)`` tuple. All of them are float32 and tightly
packed, with the channel counts NGX expects: color 4, depth 1, motion vectors 2,
normals 4 (roughness packed into ``.w``), albedos 4, output 4.

NGX reads through CUDA texture objects and writes through a surface object, so
the denoiser owns the CUDA arrays and blits in and out of them, the way Mitsuba's
own DLSS denoiser does. It needs neither the CUDA toolkit nor any particular
rendering framework; :mod:`denoiser.dlss.drjit` wires it up to Dr.Jit tensors.

DLSS-D is *temporal*: it wants one sample per pixel with a known sub-pixel jitter,
screen-space motion vectors and separated guide buffers, not an accumulated image.
Reset the history whenever the camera or the scene jumps.
"""

import ctypes

from . import _cuda, _download
from ._feature import (Feature, FEATURE_RAY_RECONSTRUCTION, INPUTS, MIN_EXTENT, QUALITY)
from ._ngx import (Common, CUDADevice, DLSSError, Driver, FeatureRequirement, Parameters,
                   RESULT_SUCCESS, VERSION_API, check, driver_available, result_name,
                   support_reason)
from ._ngx import application_id as _application_id, data_path as _data_path

__all__ = ["DLSSDenoiser", "DLSSError", "QUALITY", "is_available", "support_status",
           "driver_available"]


def support_status(*, context: int | None = None, library_path=None, download: bool = True,
                   version: str | None = None, application_id=None,
                   data_path=None) -> tuple[bool, str]:
    """Ask the driver whether Ray Reconstruction runs here, as (supported, reason).

    This needs no NGX initialisation. The query is per device, so it uses
    `context`, or whichever CUDA context is current on this thread.
    """
    try:
        driver = Driver.get("CUDA")
        path = _download.resolve(library_path, download=download, version=version)
        common = Common(path, verbose=False)
        info = common.discovery(_application_id(application_id), _data_path(data_path))
        requirement = FeatureRequirement()
        device = _cuda.context_device(context if context is not None
                                      else _cuda.current_context())
        rv = driver.GetFeatureRequirements(device, ctypes.byref(info), ctypes.byref(requirement))
        if rv != RESULT_SUCCESS:
            return False, f"GetFeatureRequirements: {result_name(rv)}"
        if requirement.FeatureSupported != 0:
            return False, support_reason(requirement)
        return True, "supported"
    except (DLSSError, _cuda.CudaError, _download.DownloadError) as e:
        return False, str(e)


def is_available(**kwargs) -> bool:
    """Whether DLSS Ray Reconstruction can be used on this system."""
    return support_status(**kwargs)[0]


class DLSSDenoiser(Feature):
    """The NVIDIA DLSS Ray Reconstruction denoiser, on raw CUDA memory.

    NGX runs on one CUDA context and stream, fixed when the denoiser is built:
    `context` defaults to the one current on this thread and `stream` to the
    default stream. Pass the renderer's own pair to have the copies and the
    evaluation ordered against its kernels without a synchronisation point.

    Beyond the feature configuration of :class:`denoiser.dlss._feature.Feature`:

    Args:
        context: The CUDA context to run on.
        stream: The CUDA stream to queue the copies and the evaluation on.
        synchronize: Wait for that stream at the end of every call.
        library_path: Directory holding the Ray Reconstruction snippet. By
            default it is taken from ``MI_DLSS_LIBRARY_PATH``, or downloaded.
        download: Fetch the snippet from https://github.com/NVIDIA/DLSS when it
            is not configured otherwise.
        version: Tag of that repository to download, or ``"latest"``.
        verbose: Forward NGX's own log messages to stdout.
    """

    def __init__(self, input_size, output_size=None, quality: str | int = "high", *,
                 context: int | None = None, stream: int = 0, synchronize: bool = False,
                 library_path=None, download: bool = True, version: str | None = None,
                 application_id=None, data_path=None, verbose: bool = False, **feature):
        super().__init__(input_size, output_size, quality, **feature)
        self.synchronize = synchronize
        self.context = context if context is not None else _cuda.current_context()
        self.stream = stream

        supported, reason = support_status(context=self.context, library_path=library_path,
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
        self._driver = Driver.get("CUDA")
        self._device = CUDADevice(cudaContext=ctypes.c_void_p(self.context),
                                  cudaStream=ctypes.c_void_p(self.stream))
        self._feature = ctypes.c_void_p()
        self._arrays: dict[str, _cuda.Array] = {}

        with _cuda.ContextGuard(self.context):
            check(self._driver.Init(_application_id(application_id), _data_path(data_path),
                                    ctypes.byref(self._device), VERSION_API,
                                    ctypes.byref(self._common.info)),
                  "NVSDK_NGX_CUDA_Init_Ext1")
            try:
                self._create_feature()
            except Exception:
                self._shutdown()
                raise

    # -- NGX lifecycle ------------------------------------------------------

    def _parameters(self) -> Parameters:
        out = ctypes.c_void_p()
        check(self._driver.AllocateParameters(ctypes.byref(out)), "AllocateParameters")
        return Parameters(out.value)

    def _create_feature(self):
        params = self._parameters()
        self.fill_create_params(params)
        rv = self._driver.CreateFeature(ctypes.byref(self._device), FEATURE_RAY_RECONSTRUCTION,
                                        ctypes.c_void_p(params.ptr), ctypes.byref(self._feature))
        self._driver.DestroyParameters(ctypes.c_void_p(params.ptr))
        check(rv, "NVSDK_NGX_CUDA_CreateFeature1")

    def _shutdown(self):
        remaining = ctypes.c_uint32(0)
        self._driver.Shutdown(ctypes.byref(self._device), ctypes.byref(remaining))

    def release(self):
        """Destroy the NGX feature and the CUDA arrays it reads and writes."""
        if not self._feature:
            return
        with _cuda.ContextGuard(self.context):
            self._driver.ReleaseFeature(self._feature)
            self._feature = ctypes.c_void_p()
            for array in self._arrays.values():
                array.destroy()
            self._arrays.clear()
            self._shutdown()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # -- evaluation ---------------------------------------------------------

    def _array(self, name: str, channels: int, size) -> _cuda.Array:
        array = self._arrays.get(name)
        if array is None:
            array = self._arrays[name] = _cuda.Array(size[0], size[1], channels)
        return array

    def __call__(self, *, color, output, depth, motion_vectors, normals, diffuse_albedo,
                 specular_albedo, roughness=None, specular_motion_vectors=None,
                 jitter=(0.0, 0.0), mv_scale=(1.0, 1.0), reset: bool = False,
                 pre_exposure: float = 1.0, exposure_scale: float = 1.0,
                 extra: dict | None = None):
        """Run Ray Reconstruction, writing the result back into `output`.

        `jitter` is the sub-pixel camera offset this frame was rendered with,
        `mv_scale` converts the motion vector buffer into pixels, and `reset`
        drops the temporal history (it is implied for the first call). Anything
        else DLSS-D accepts goes in `extra`, keyed by NGX parameter name.
        """
        if not self._feature:
            raise DLSSError("this denoiser has been released")

        def resolve(value, name, channels):
            if channels is None:  # from extra, the caller knows its layout
                return _cuda.image_source(value, name)
            return _cuda.reshape(value, *self.input_size, channels, name)

        inputs = self.input_names({
            "color": color, "depth": depth, "motion_vectors": motion_vectors,
            "normals": normals, "roughness": roughness, "diffuse_albedo": diffuse_albedo,
            "specular_albedo": specular_albedo,
            "specular_motion_vectors": specular_motion_vectors}, extra, resolve)
        destination = _cuda.reshape(output, *self.output_size, 4, "output")

        with _cuda.ContextGuard(self.context):
            out_array = self._array("Output", 4, self.output_size)
            arrays = {}
            for name, source in inputs.items():
                if (shared := next((n for n, s in inputs.items()
                                    if s is source and n in arrays), None)) is not None:
                    arrays[name] = arrays[shared]  # e.g. roughness packed into normals.w
                    continue
                arrays[name] = self._array(name, source.channels, self.input_size)
                arrays[name].upload(source, self.stream)

            params = self._parameters()
            for name, array in arrays.items():
                params.set_ptr(name, array.texture_ptr())
            params.set_ptr("Output", out_array.surface_ptr())
            self.fill_eval_params(params, jitter, mv_scale, reset or self._reset,
                                  pre_exposure, exposure_scale)
            rv = self._driver.EvaluateFeature(self._feature, ctypes.c_void_p(params.ptr), None)
            self._driver.DestroyParameters(ctypes.c_void_p(params.ptr))
            check(rv, "NVSDK_NGX_CUDA_EvaluateFeature")

            out_array.download(destination, self.stream)
            if self.synchronize:
                _cuda.check(_cuda.driver().StreamSynchronize(ctypes.c_void_p(self.stream)),
                            "cuStreamSynchronize")
        self._reset = False
        return output

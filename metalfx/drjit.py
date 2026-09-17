"""The MetalFX scalers on Dr.Jit tensors, running on Dr.Jit's own Metal device.

Everything Dr.Jit-specific lives here: where Dr.Jit runs, and the conversion
between its tensors and the textures MetalFX reads.
:mod:`rendering_denoisers.metalfx.metal` underneath knows nothing about it.

``drjit-core`` exports the handles of its Metal backend the same way it exports
its CUDA context and stream, so the effects can be built on the device and the
command queue the renderer already uses -- no second device, no cross-device
copies, and the work is encoded onto the queue the kernels are on::

    from rendering_denoisers.metalfx.drjit import TemporalDenoisedScaler

    fx = TemporalDenoisedScaler((960, 540), (1920, 1080))
    denoised = fx(color=color, depth=depth, motion=motion, normal=normal,
                  diffuse_albedo=diffuse, specular_albedo=specular,
                  roughness=roughness)          # mi.TensorXf in, mi.TensorXf out

One limitation to be aware of: the pixels still make a round trip through the
host. Dr.Jit exposes its Metal *device* and *queue*, but nothing that reaches the
``MTLBuffer`` behind a variable, so there is no way from Python to alias one as a
texture. Should Dr.Jit gain a buffer accessor or DLPack on Metal, the one place
to change is :class:`rendering_denoisers.metalfx._objc.Texture`; everything
here stays.
"""

import ctypes

import drjit as dr

from . import metal as _metal
from ._objc import MetalFXError, as_object

__all__ = ["SpatialScaler", "TemporalScaler", "TemporalDenoisedScaler", "MetalFXError",
           "device", "queue", "is_available"]


# drjit-core declares these with C++ linkage, so the exported name is mangled.
# The two candidates cover MSVC and the Itanium ABI; the plain name is tried
# first, in case a future build marks them extern "C".
_SYMBOLS = {
    "jit_metal_device_handle": ("?jit_metal_device_handle@@YAPEAXXZ",
                                "_Z23jit_metal_device_handlev"),
    "jit_metal_queue":         ("?jit_metal_queue@@YAPEAXXZ", "_Z15jit_metal_queuev"),
}


class _Core:
    """The drjit-core entry points that say which Metal device Dr.Jit is on."""

    _instance = None

    def __init__(self):
        name = "libdrjit-core.dylib"
        try:
            # Dr.Jit has already loaded it, next to the package in a wheel or
            # elsewhere in a build tree; asked for by its bare name, the loader
            # hands back that copy, so this only resolves symbols in it.
            self.lib = ctypes.CDLL(name)
        except OSError as e:
            raise MetalFXError(f"could not load {name}: {e}") from e
        self.device = self._bind("jit_metal_device_handle")
        self.queue = self._bind("jit_metal_queue")

    def _bind(self, name):
        for candidate in (name, *_SYMBOLS[name]):
            if (fn := getattr(self.lib, candidate, None)) is not None:
                fn.restype, fn.argtypes = ctypes.c_void_p, []
                return fn
        raise MetalFXError(f"drjit-core does not export {name}; this Dr.Jit was built "
                           f"without the Metal backend")

    @classmethod
    def get(cls) -> "_Core":
        if cls._instance is None:
            cls._instance = _Core()
        return cls._instance


def device():
    """The MTLDevice Dr.Jit renders on."""
    handle = _Core.get().device()
    if not handle:
        raise MetalFXError("Dr.Jit has no Metal device; is a metal_* variant in use?")
    return as_object(handle)


def queue():
    """The MTLCommandQueue Dr.Jit submits its kernels on."""
    handle = _Core.get().queue()
    if not handle:
        raise MetalFXError("Dr.Jit has no Metal command queue")
    return as_object(handle)


def is_available() -> bool:
    """Whether Dr.Jit is running on Metal and the effects can share its device."""
    try:
        return bool(dr.has_backend(dr.JitBackend.Metal)) and _Core.get().device() is not None
    except Exception:
        return False


class _DrJit:
    """Builds on Dr.Jit's device and queue, and hands back Dr.Jit tensors."""

    def __init__(self, *args, **kwargs):
        # Fall back to the system device when Dr.Jit is not on Metal -- a Dr.Jit
        # LLVM build still has pixels to scale, they just come from the host.
        if "device" not in kwargs and is_available():
            kwargs["device"] = device()
            kwargs.setdefault("queue", queue())
        super().__init__(*args, **kwargs)

    def __call__(self, output=None, **buffers):
        tensor = next((type(v) for v in buffers.values()
                       if v is not None and hasattr(v, "array")), None)
        result = super().__call__(output=output, **buffers)
        if tensor is None or output is not None:
            return result
        return tensor(result)


class SpatialScaler(_DrJit, _metal.SpatialScaler):
    __doc__ = _metal.SpatialScaler.__doc__


class TemporalScaler(_DrJit, _metal.TemporalScaler):
    __doc__ = _metal.TemporalScaler.__doc__


class TemporalDenoisedScaler(_DrJit, _metal.TemporalDenoisedScaler):
    __doc__ = _metal.TemporalDenoisedScaler.__doc__

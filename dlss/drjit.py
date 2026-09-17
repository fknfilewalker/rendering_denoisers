"""DLSS Ray Reconstruction on Dr.Jit tensors, i.e. on Mitsuba renderings.

Everything Dr.Jit-specific lives here: where Dr.Jit runs (its CUDA context and
stream, read out of ``drjit-core``) and how a rendering is packed into the
buffers NGX wants. :mod:`rendering_denoisers.dlss.cuda` underneath knows
nothing about it.

The interface follows the ``DLSSDenoiser`` proposed in mitsuba3 PR #1957, so that
it can be used from an ordinary ``pip install mitsuba`` without rebuilding
anything. Images are ``mi.TensorXf`` shaped ``(height, width, channels)``::

    from rendering_denoisers.dlss.drjit import DLSSDenoiser

    dlss = DLSSDenoiser((960, 540), (1920, 1080), quality="high")
    denoised = dlss(noisy, albedo, normals, depth, specular_albedo=specular,
                    roughness=roughness, flow=motion, to_sensor=world_to_camera,
                    jitter=jitter, reset=camera_moved)

Because the copies and the NGX evaluation are queued on Dr.Jit's own stream, they
are ordered against the renderer's kernels without a synchronisation point.

DLSS-D is temporal: feed it single frames of a sequence, typically one sample per
pixel, each with a different sub-pixel jitter -- not an accumulation buffer. The
first frames of a sequence stay noticeably noisier than later ones, and the
history has to be dropped with ``reset=True`` whenever the sequence breaks.
"""

import ctypes
import importlib
import sys

import drjit as dr

from . import cuda as _backend
from ._ngx import DLSSError, driver_available
from .cuda import QUALITY

__all__ = ["DLSSDenoiser", "DLSSError", "QUALITY", "context", "stream", "is_available",
           "support_status", "driver_available"]


# ---------------------------------------------------------------------------
# where Dr.Jit runs
# ---------------------------------------------------------------------------

# drjit-core declares these with C++ linkage, so the exported name is mangled.
# The two candidates cover MSVC and the Itanium ABI; the plain name is tried
# first, in case a future build marks them extern "C".
_SYMBOLS = {
    "jit_cuda_context":    ("?jit_cuda_context@@YAPEAXXZ", "_Z16jit_cuda_contextv"),
    "jit_cuda_stream":     ("?jit_cuda_stream@@YAPEAXXZ", "_Z15jit_cuda_streamv"),
    "jit_cuda_device_raw": ("?jit_cuda_device_raw@@YAHXZ", "_Z19jit_cuda_device_rawv"),
}


class _Core:
    """The handful of drjit-core entry points that say where Dr.Jit is running."""

    _instance = None

    def __init__(self):
        name = "drjit-core.dll" if sys.platform == "win32" else "libdrjit-core.so"
        try:
            # Dr.Jit has already loaded it, next to the package in a wheel or
            # elsewhere in a build tree; asked for by its bare name, the loader
            # hands back that copy, so this only resolves symbols in it.
            self.lib = ctypes.CDLL(name)
        except OSError as e:
            raise DLSSError(f"could not load {name}: {e}") from e
        self.context = self._bind("jit_cuda_context", ctypes.c_void_p)
        self.stream = self._bind("jit_cuda_stream", ctypes.c_void_p)
        self.device_raw = self._bind("jit_cuda_device_raw", ctypes.c_int)

    def _bind(self, name, restype):
        for candidate in (name, *_SYMBOLS[name]):
            if (fn := getattr(self.lib, candidate, None)) is not None:
                fn.restype, fn.argtypes = restype, []
                return fn
        raise DLSSError(f"drjit-core does not export {name}")

    @classmethod
    def get(cls) -> "_Core":
        if cls._instance is None:
            cls._instance = _Core()
        return cls._instance


def context() -> int:
    """The CUDA context Dr.Jit renders in."""
    value = _Core.get().context()
    if not value:
        raise DLSSError("Dr.Jit has no CUDA context; is a cuda_* variant in use?")
    return value


def stream() -> int:
    """The CUDA stream Dr.Jit launches its kernels on."""
    return _Core.get().stream() or 0


def support_status(**kwargs) -> tuple[bool, str]:
    """Whether Ray Reconstruction runs on the device Dr.Jit renders on.

    Returns (supported, reason); see
    :func:`rendering_denoisers.dlss.cuda.support_status`.
    """
    kwargs.setdefault("context", context())
    return _backend.support_status(**kwargs)


def is_available(**kwargs) -> bool:
    """Whether DLSS Ray Reconstruction can be used with this Dr.Jit backend."""
    try:
        return support_status(**kwargs)[0]
    except DLSSError:
        return False  # Dr.Jit is not on CUDA at all


# ---------------------------------------------------------------------------
# tensor plumbing
# ---------------------------------------------------------------------------

def _types(tensor):
    """The (TensorXf, Float, UInt32, Array3f) of whichever variant the caller uses."""
    Float = type(tensor.array)
    module = importlib.import_module(Float.__module__)
    return type(tensor), Float, dr.uint32_array_t(Float), module.Array3f


def _flat(tensor, Float):
    """The detached contents of a tensor as a fresh flat array.

    Gradients cannot survive the round trip through NGX and the raw device
    pointers below would silently bypass them, so everything is detached. The
    copy is of the Python handle rather than of the buffer, and it keeps
    ``dr.make_opaque`` from rebinding a variable the caller still holds.
    """
    return Float(dr.detach(tensor).array)


def _pack(array, channels: int, count: int, Float, UInt32, fill=None):
    """Re-interleave a flat (count * n) buffer as (count * `channels`).

    Channels beyond the source are left at zero, or at `fill` when given.
    """
    n = len(array) // count if count else 0
    if n == channels and fill is None:
        return Float(array)
    index = dr.arange(UInt32, count)
    out = dr.zeros(Float, count * channels) if fill is None \
        else dr.full(Float, fill, count * channels)
    for i in range(min(n, channels)):
        dr.scatter(out, dr.gather(Float, array, index * n + i), index * channels + i)
    return out


# ---------------------------------------------------------------------------
# the denoiser
# ---------------------------------------------------------------------------

class DLSSDenoiser(_backend.DLSSDenoiser):
    """Wrapper for the NVIDIA DLSS Ray Reconstruction denoiser, on Mitsuba types.

    In contrast to ``mi.OptixDenoiser``, this denoiser is inherently temporal: it
    expects a sequence of independently rendered frames (typically a single
    sample per pixel each), the sub-pixel jitter applied to the camera of every
    frame, and screen-space motion vectors relating consecutive frames.

    Args:
        input_size: ``(width, height)`` of the noisy images fed to the denoiser.
        output_size: ``(width, height)`` of the denoised images. When it is
            larger than `input_size`, DLSS upscales. Defaults to `input_size`,
            i.e. no upscaling (DLAA mode).
        quality: Quality/performance tradeoff of the upscaling step, one of
            ``"high"``, ``"balanced"`` or ``"fast"``. Has no effect when
            `output_size` equals `input_size`.

    Everything else is forwarded to
    :class:`rendering_denoisers.dlss.cuda.DLSSDenoiser`, notably
    ``library_path``, ``download`` and ``version``, which control where the Ray
    Reconstruction snippet comes from.
    """

    def __init__(self, input_size, output_size=None, quality: str = "high", **kwargs):
        kwargs.setdefault("context", context())
        kwargs.setdefault("stream", stream())
        # Mitsuba renders HDR, stores its images top to bottom, hands out linear
        # camera-space depth, and this class takes motion vectors at the input
        # resolution with the roughness packed into normals.w.
        kwargs.setdefault("hdr", True)
        kwargs.setdefault("mv_low_res", True)
        kwargs.setdefault("hw_depth", False)
        kwargs.setdefault("packed_roughness", True)
        super().__init__(input_size, output_size, quality, **kwargs)

    def __call__(self, noisy, albedo, normals, depth, specular_albedo=None, roughness=None,
                 flow=None, specular_flow=None, to_sensor=None, jitter=(0.0, 0.0),
                 reset: bool = False):
        """Denoise one frame of a sequence.

        Every tensor is shaped ``(height, width, channels)`` at `input_size`.

        Args:
            noisy: The noisy input, 3 or 4 channels. A fourth channel is passed
                through rather than denoised.
            albedo: Diffuse albedo, 3 channels, clamped to [0, 1].
            normals: Shading normals, 3 channels, in the frame `flow` and `depth`
                were computed in (typically world space); `to_sensor` transforms
                them.
            depth: Linear camera-space depth, 1 channel -- the distance of the
                shading point to the plane of the sensor, not to the sensor
                itself.
            specular_albedo: Specular albedo, 3 channels. Leaving it out degrades
                the quality of reflections.
            roughness: Surface roughness, 1 channel.
            flow: Screen-space motion vectors in input-resolution pixels,
                2 channels, pointing from a pixel in this frame to where it was
                in the previous one. Leaving it out produces ghosting whenever
                the camera or the scene moves.
            specular_flow: Motion vectors of the virtual image seen in specular
                surfaces, 2 channels, in the same units as `flow`.
            to_sensor: A transform applied to `normals` before denoising, e.g. an
                ``mi.Transform4f`` holding the world-to-camera frame.
            jitter: The sub-pixel offset the camera was rendered with, in pixels
                relative to the centre of a pixel. Every frame of a sequence must
                use a different one, e.g. taken from a low-discrepancy sequence
                covering [-0.5, 0.5]^2.
            reset: Discard the history accumulated from the previous frames. Set
                this whenever the frame is not temporally related to the previous
                one. Implied for the first call.

        Returns:
            The denoised frame at `output_size`, with the same number of channels
            as `noisy`.
        """
        Tensor, Float, UInt32, Array3f = _types(noisy)
        width, height = self.input_size
        out_width, out_height = self.output_size
        n_in, n_out = width * height, out_width * out_height
        channels = noisy.shape[2]
        if channels not in (3, 4):
            raise DLSSError(f"the noisy input must have 3 or 4 channels, got {channels}")
        self._check(noisy=(noisy, channels), albedo=(albedo, 3), normals=(normals, 3),
                    depth=(depth, 1), specular_albedo=(specular_albedo, 3),
                    roughness=(roughness, 1), flow=(flow, 2), specular_flow=(specular_flow, 2))

        noisy_array = _flat(noisy, Float)
        color = _pack(noisy_array, 4, n_in, Float, UInt32, fill=None if channels == 4 else 1.0)
        diffuse = _pack(dr.clip(_flat(albedo, Float), 0.0, 1.0), 4, n_in, Float, UInt32, fill=1.0)
        # Unspecified specular albedo is black, not white: telling DLSS the whole
        # image is a mirror would wreck it. The fourth channel is 1 either way.
        specular = _pack(dr.clip(_flat(specular_albedo, Float), 0.0, 1.0)
                         if specular_albedo is not None else dr.zeros(Float, n_in * 3),
                         4, n_in, Float, UInt32, fill=1.0)
        normal_roughness = self._normals(normals, roughness, to_sensor, n_in, Float, UInt32,
                                         Array3f)
        depth_array = _flat(depth, Float)
        motion = _flat(flow, Float) if flow is not None else dr.zeros(Float, n_in * 2)
        specular_motion = _flat(specular_flow, Float) if specular_flow is not None \
            else dr.zeros(Float, n_in * 2)
        output = dr.empty(Float, n_out * 4)

        # Every buffer needs real device memory before it can be handed to DLSS.
        # dr.eval() alone is not enough: a zero-initialised array is a literal
        # constant and stays one, without ever receiving a data pointer.
        dr.make_opaque(color, diffuse, specular, normal_roughness, depth_array, motion,
                       specular_motion, output)

        super().__call__(color=color, output=output, depth=depth_array, motion_vectors=motion,
                         normals=normal_roughness, diffuse_albedo=diffuse,
                         specular_albedo=specular, specular_motion_vectors=specular_motion,
                         jitter=jitter, reset=reset)

        result = _pack(output, channels, n_out, Float, UInt32)
        if channels == 4:
            # DLSS does not upscale the alpha channel, take it from the nearest
            # pixel of the noisy input instead.
            index = dr.arange(UInt32, n_out)
            x, y = index % out_width, index // out_width
            source = (y * height // out_height) * width + (x * width // out_width)
            dr.scatter(result, dr.gather(Float, noisy_array, source * 4 + 3), index * 4 + 3)
        return Tensor(result, (out_height, out_width, channels))

    def _normals(self, normals, roughness, to_sensor, n_in, Float, UInt32, Array3f):
        """Transform the normals into the requested frame, roughness into .w.

        Only the linear part of `to_sensor` is applied: an ``mi.Transform4f``
        matrix-multiplied with a bare ``Array3f`` takes the ``Point3f`` overload
        and would translate the normals. That is the exact normal transform for
        the rigid world-to-camera frames this is used with; pass a callable to do
        something else, e.g. ``lambda n: to_sensor @ mi.Normal3f(n)`` for the
        inverse transpose.
        """
        array = _flat(normals, Float)
        index = dr.arange(UInt32, n_in)
        if to_sensor is not None:
            n = Array3f(*(dr.gather(Float, array, index * 3 + i) for i in range(3)))
            if callable(to_sensor):
                n = to_sensor(n)
            elif (m := getattr(to_sensor, "matrix", None)) is not None:
                n = Array3f(*(m[r][0] * n.x + m[r][1] * n.y + m[r][2] * n.z for r in range(3)))
            else:
                n = to_sensor @ n
            array = dr.zeros(Float, n_in * 3)
            for i in range(3):
                dr.scatter(array, n[i], index * 3 + i)
        packed = _pack(array, 4, n_in, Float, UInt32)
        if roughness is not None:
            dr.scatter(packed, _flat(roughness, Float), index * 4 + 3)
        return packed

    def _check(self, **tensors):
        width, height = self.input_size
        for name, (tensor, channels) in tensors.items():
            if tensor is None:
                continue
            if len(tensor.shape) != 3:
                raise DLSSError(f"the {name} must be a (height, width, channels) tensor, "
                                f"got shape {tensor.shape}")
            if (tensor.shape[1], tensor.shape[0]) != (width, height):
                raise DLSSError(f"the {name} must have the input size {width} x {height}, "
                                f"got {tensor.shape[1]} x {tensor.shape[0]}")
            if tensor.shape[2] != channels:
                raise DLSSError(f"the {name} must have exactly {channels} channel(s), "
                                f"got {tensor.shape[2]}")

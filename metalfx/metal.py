"""The MetalFX spatial and temporal scalers, on Metal textures or plain arrays.

Nothing here is tied to a rendering framework. An effect owns its input and
output textures and the command queue it submits on, so the only Metal object a
caller has to supply is a device -- and even that defaults to the system one.

Each buffer can be given two ways:

* as an array (numpy, a Dr.Jit tensor, a torch CPU tensor, anything with DLPack
  or the array protocol), which is copied into the texture the effect owns, or
* as an ``MTLTexture`` the caller already has, by PyObjC object or by address,
  which is bound directly and never copied. That is the path for a renderer that
  already works in Metal -- ``slangpy.Texture.native_handle`` and friends.

Usage, upscaling 960x540 to 1920x1080 with the real-time ray tracing denoiser::

    from rendering_denoisers.metalfx.metal import TemporalDenoisedScaler

    fx = TemporalDenoisedScaler((960, 540), (1920, 1080))
    fx.world_to_view = camera.world_to_view      # 4x4, row major
    fx.view_to_clip = camera.view_to_clip
    fx.jitter = (jx, jy)
    denoised = fx(color=color, depth=depth, motion=motion, normal=normal,
                  diffuse_albedo=diffuse, specular_albedo=specular,
                  roughness=roughness)

The scalers are temporal: they want one sample per pixel at a known sub-pixel
jitter, screen-space motion vectors in pixels pointing from this frame to the
previous one, and separated guide buffers -- not an accumulated image. Set
``reset`` whenever the camera or the scene jumps.
"""

import numpy as np

from . import _objc
from ._objc import MetalFXError, Texture, as_object, camel, default_device, to_simd

try:
    import MetalFX as _MetalFX
except ImportError as e:  # pragma: no cover - the package only loads on macOS
    raise ImportError("rendering_denoisers.metalfx needs macOS with "
                      "pyobjc-framework-MetalFX") from e

__all__ = ["Effect", "SpatialScaler", "TemporalScaler", "TemporalDenoisedScaler",
           "MetalFXError", "is_available"]


def is_available() -> bool:
    """Whether MetalFX can be used on this machine."""
    try:
        default_device()
        return True
    except Exception:
        return False


class Effect:
    """Base class: allocates the textures an MTLFX effect binds, and runs it."""

    DESCRIPTOR: type
    FACTORY: str
    INPUTS: dict[str, str] = {}

    def __init__(self, input_size, output_size=None, *, device=None, queue=None,
                 output_format: str = "rgba16_float"):
        self.device = as_object(device) if device is not None else default_device()
        self.queue = as_object(queue) if queue is not None else self.device.newCommandQueue()
        self.output_format = output_format
        self.textures: dict[str, Texture] = {}
        self.resize(input_size, output_size or input_size)

    def __getattr__(self, name: str) -> Texture:
        textures = self.__dict__.get("textures", {})
        if name in textures:
            return textures[name]
        raise AttributeError(name)

    @property
    def input_size(self):
        return (self.width, self.height)

    @property
    def output_size(self):
        return (self.output_width, self.output_height)

    def resize(self, input_size, output_size=None):
        self.width, self.height = (int(v) for v in input_size)
        self.output_width, self.output_height = (int(v) for v in (output_size or input_size))

        self.textures = {name: Texture(self.device, self.width, self.height, fmt, label=name)
                         for name, fmt in self.INPUTS.items()}
        self.textures["output"] = Texture(self.device, self.output_width, self.output_height,
                                          self.output_format, label="output")

        descriptor = self.DESCRIPTOR.alloc().init()
        descriptor.setInputWidth_(self.width)
        descriptor.setInputHeight_(self.height)
        descriptor.setOutputWidth_(self.output_width)
        descriptor.setOutputHeight_(self.output_height)
        for name, texture in self.textures.items():
            getattr(descriptor, f"set{camel(name)}TextureFormat_")(texture.pixel_format)
        self.configure(descriptor)

        self.effect = getattr(descriptor, self.FACTORY)(self.device)
        if self.effect is None:
            raise MetalFXError(f"{type(self).__name__}: unsupported configuration "
                               f"{self.width}x{self.height} -> "
                               f"{self.output_width}x{self.output_height}")
        self.effect.setInputContentWidth_(self.width)
        self.effect.setInputContentHeight_(self.height)
        self.bind()

    def bind(self, **textures: Texture):
        """Point the effect at its textures, replacing any given by name."""
        self.textures.update(textures)
        for name, texture in self.textures.items():
            getattr(self.effect, f"set{camel(name)}Texture_")(texture.texture)

    def configure(self, descriptor):
        """Hook for subclasses to set descriptor properties before creation."""

    def update(self):
        """Hook for subclasses to push per-frame properties onto the effect."""

    def _stage(self, buffers: dict):
        """Bind the caller's textures, or copy the caller's arrays into ours."""
        adopted, external = {}, False
        for name, value in buffers.items():
            if value is None:
                continue
            if name not in self.textures:
                raise MetalFXError(f"{type(self).__name__} has no {name!r} input; "
                                   f"expected {sorted(self.INPUTS)}")
            mine = self.textures[name]
            if _is_texture(value):
                size = (mine.width, mine.height)
                adopted[name] = Texture(self.device, *size, mine.format, texture=value,
                                        label=name)
                external = True
            else:
                mine.write(value, name)
        if external:
            self.bind(**adopted)

    def __call__(self, output=None, **buffers):
        """Run the effect over `buffers`, returning the result.

        With `output` given as a texture the result is written there and that
        texture is returned; otherwise it is read back into a numpy array.
        """
        if output is not None and _is_texture(output):
            buffers = dict(buffers, output=output)
        self._stage(buffers)
        self.execute()
        if output is not None and _is_texture(output):
            return output
        return self.textures["output"].read(output)

    def execute(self):
        """Encode and run the effect, blocking until it is done."""
        self.update()
        command_buffer = self.queue.commandBuffer()
        self.effect.encodeToCommandBuffer_(command_buffer)
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        if command_buffer.error() is not None:
            raise MetalFXError(f"{type(self).__name__}: {command_buffer.error()}")


def _is_texture(value) -> bool:
    """Whether `value` is an MTLTexture the caller owns, rather than pixel data."""
    if isinstance(value, Texture):
        return True
    if isinstance(value, (int, np.integer)):
        return True  # an address
    return hasattr(value, "pixelFormat")  # a PyObjC MTLTexture


class SpatialScaler(Effect):
    """Single-frame upscaler: no history, no guide buffers."""

    DESCRIPTOR = _MetalFX.MTLFXSpatialScalerDescriptor
    FACTORY = "newSpatialScalerWithDevice_"
    INPUTS = {"color": "rgba16_float"}

    def __init__(self, *args, hdr: bool = True, **kwargs):
        self.hdr = hdr
        super().__init__(*args, **kwargs)

    def configure(self, descriptor):
        descriptor.setColorProcessingMode_(
            _MetalFX.MTLFXSpatialScalerColorProcessingModeHDR if self.hdr
            else _MetalFX.MTLFXSpatialScalerColorProcessingModePerceptual)


class TemporalScaler(Effect):
    """Temporal upscaler: accumulates across frames using motion vectors."""

    DESCRIPTOR = _MetalFX.MTLFXTemporalScalerDescriptor
    FACTORY = "newTemporalScalerWithDevice_"
    # r32_float depth so compute shaders can write it; motion in pixels, current -> previous
    INPUTS = {"color": "rgba16_float", "depth": "r32_float", "motion": "rg16_float"}

    def __init__(self, *args, **kwargs):
        self.jitter = (0.0, 0.0)              # sub-pixel offset in [-0.5, 0.5]
        self.motion_vector_scale = (1.0, 1.0)
        self.depth_reversed = False
        super().__init__(*args, **kwargs)

    def resize(self, *args, **kwargs):
        self.reset = True
        super().resize(*args, **kwargs)

    def update(self):
        self.effect.setJitterOffsetX_(float(self.jitter[0]))
        self.effect.setJitterOffsetY_(float(self.jitter[1]))
        self.effect.setMotionVectorScaleX_(float(self.motion_vector_scale[0]))
        self.effect.setMotionVectorScaleY_(float(self.motion_vector_scale[1]))
        self.effect.setDepthReversed_(bool(self.depth_reversed))
        self.effect.setReset_(bool(self.reset))

    def execute(self):
        super().execute()
        self.reset = False


class TemporalDenoisedScaler(TemporalScaler):
    """Real-time ray tracing denoiser and temporal upscaler (macOS 26+)."""

    DESCRIPTOR = _MetalFX.MTLFXTemporalDenoisedScalerDescriptor
    FACTORY = "newTemporalDenoisedScalerWithDevice_"
    # normal: world space
    INPUTS = TemporalScaler.INPUTS | {
        "diffuse_albedo": "rgba16_float",
        "specular_albedo": "rgba16_float",
        "normal": "rgba16_float",
        "roughness": "r16_float",
    }

    def __init__(self, *args, **kwargs):
        self.world_to_view = np.eye(4, dtype=np.float32)
        self.view_to_clip = np.eye(4, dtype=np.float32)
        super().__init__(*args, **kwargs)

    def update(self):
        super().update()
        self.effect.setWorldToViewMatrix_(to_simd(self.world_to_view))
        self.effect.setViewToClipMatrix_(to_simd(self.view_to_clip))

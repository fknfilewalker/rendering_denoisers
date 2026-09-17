"""The MetalFX spatial and temporal scalers, driven through PyObjC.

MetalFX upscales and, since macOS 26, denoises ray traced frames. Nothing here is
tied to a rendering framework: an effect owns its Metal textures and command
queue, so a caller only supplies pixels.

Pick a backend by how the pixels reach the effect:

* :mod:`denoiser.metalfx.metal` -- Metal, and anything that can produce a CPU
  array. An ``MTLTexture`` you already own (by PyObjC object or by address, which
  is what ``slangpy.Texture.native_handle`` hands out) is bound directly and
  never copied; a numpy, Dr.Jit or torch CPU array is copied into the texture the
  effect owns.
* :mod:`denoiser.metalfx.drjit` -- Dr.Jit tensors in and out, built on the Metal
  device and command queue Dr.Jit itself renders with, so the effect runs on the
  renderer's queue rather than a second one.

Each exposes the same three effects and an ``is_available()``::

    from denoiser.metalfx.drjit import TemporalDenoisedScaler, is_available

Requirements: macOS on Apple silicon, with pyobjc-framework-Metal and
pyobjc-framework-MetalFX. The denoised temporal scaler needs macOS 26.
"""

from ._objc import FORMATS, MetalFXError

__all__ = ["MetalFXError", "FORMATS"]

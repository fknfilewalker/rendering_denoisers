"""Real-time denoisers and upscalers for path traced renderings.

Two vendor libraries, one per platform, of which only the one this machine can
actually run is imported:

* :mod:`rendering_denoisers.dlss` -- NVIDIA DLSS Ray Reconstruction, on Windows
  and Linux. Backends for raw CUDA memory, Vulkan images, and Dr.Jit tensors
  (:mod:`rendering_denoisers.dlss.drjit`, the one Mitsuba renderings go
  through). Needs an RTX GPU and driver 590 or newer; the feature library is
  downloaded on first use.
* :mod:`rendering_denoisers.metalfx` -- the MetalFX spatial and temporal
  scalers, including the denoised temporal scaler of macOS 26. Needs macOS and
  pyobjc.

Both are temporal: they want one sample per pixel with a known sub-pixel jitter,
screen-space motion vectors and separated guide buffers, not an accumulated
image, and their history has to be reset whenever the camera or the scene jumps.
"""

import sys

# NGX ships only for Windows and Linux, MetalFX only for Apple platforms, so on
# any given machine exactly one of the two is the denoiser. Importing it is
# cheap: neither touches the GPU until a denoiser is built.
if sys.platform == "darwin":
    from . import metalfx
    __all__ = ["metalfx"]
else:
    from . import dlss
    __all__ = ["dlss"]

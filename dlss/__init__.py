"""NVIDIA DLSS Ray Reconstruction, driven from Python through ctypes.

DLSS Ray Reconstruction (DLSS-D) is a real-time denoiser for ray tracing which
additionally performs temporal antialiasing and optional upscaling. Nothing here
has to be compiled: the NGX API of the display driver is bound with ctypes, and
the Ray Reconstruction snippet is downloaded from NVIDIA's repository on first
use.

Pick a backend by how the pixels reach the denoiser:

* :mod:`denoiser.dlss.cuda` -- raw CUDA memory, staged through CUDA arrays. Works
  with anything that can hand out a device pointer: Dr.Jit, torch, cupy, or a
  bare address.
* :mod:`denoiser.dlss.vulkan` -- Vulkan images, read and written in place and
  recorded into a command buffer. No copies, but the extensions it needs have to
  be enabled when the device is created.
* :mod:`denoiser.dlss.drjit` -- Dr.Jit tensors, i.e. Mitsuba renderings. Adds the
  packing of a rendering into the buffers NGX wants on top of the CUDA backend,
  and runs on Dr.Jit's own context and stream.

Each exposes a ``DLSSDenoiser`` and an ``is_available()``::

    from denoiser.dlss.drjit import DLSSDenoiser, is_available

Requirements: an RTX GPU and NVIDIA driver 590 or newer.

The snippet is downloaded on the first call, or ahead of time with
``python -m denoiser.dlss``, and cached. It is NVIDIA's binary, under the licence
of https://github.com/NVIDIA/DLSS.
"""

from ._download import DownloadError, cache_directory, clear_cache, library
from ._feature import (FLAG_ALPHA_UPSCALING, FLAG_AUTO_EXPOSURE, FLAG_DEPTH_INVERTED,
                       FLAG_IS_HDR, FLAG_MV_JITTERED, FLAG_MV_LOW_RES, MIN_EXTENT, QUALITY)
from ._ngx import DLSSError, driver_available

__all__ = ["DLSSError", "DownloadError", "QUALITY", "MIN_EXTENT", "driver_available",
           "library", "cache_directory", "clear_cache", "FLAG_IS_HDR", "FLAG_MV_LOW_RES",
           "FLAG_MV_JITTERED", "FLAG_DEPTH_INVERTED", "FLAG_AUTO_EXPOSURE",
           "FLAG_ALPHA_UPSCALING"]

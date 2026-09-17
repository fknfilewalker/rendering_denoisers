"""The Ray Reconstruction feature configuration that every backend shares.

What a DLSS-D feature *is* -- its resolution pair, quality level, creation flags
and the two parameter blocks NGX reads at create and evaluate time -- does not
depend on whether the pixels arrive as CUDA memory or as Vulkan images. That part
lives here; :mod:`denoiser.dlss.cuda` and :mod:`denoiser.dlss.vulkan` add the
resource handling around it.
"""

from ._ngx import DLSSError, Parameters

# quality -> NVSDK_NGX_PerfQuality_Value. The three names are the ones mitsuba3
# PR #1957 exposes; the constants below cover the rest of the enum, and any of
# them can be passed as a raw integer instead.
QUALITY = {"high": 2, "balanced": 1, "fast": 0}
PERF_MAX = 0
PERF_BALANCED = 1
PERF_MAX_QUALITY = 2
PERF_ULTRA_PERFORMANCE = 3
PERF_ULTRA_QUALITY = 4
PERF_DLAA = 5

# NVSDK_NGX_DLSS_Feature_Flags
FLAG_IS_HDR = 1 << 0
FLAG_MV_LOW_RES = 1 << 1
FLAG_MV_JITTERED = 1 << 2
FLAG_DEPTH_INVERTED = 1 << 3
FLAG_AUTO_EXPOSURE = 1 << 6
FLAG_ALPHA_UPSCALING = 1 << 7

FEATURE_RAY_RECONSTRUCTION = 13
MIN_EXTENT = 32  # DLSS refuses to create a feature below this resolution

# Evaluation inputs, in the order NGX names them, with the channel count each one
# carries. Backends that stage through typed memory need the counts; ones that
# take images of their own read them off the image instead.
INPUTS = {
    "color": ("Color", 4),
    "depth": ("Depth", 1),
    "motion_vectors": ("MotionVectors", 2),
    "normals": ("GBuffer.Normals", 4),
    "roughness": ("GBuffer.Roughness", 1),
    "diffuse_albedo": ("DLSS.Input.DiffuseAlbedo", 4),
    "specular_albedo": ("DLSS.Input.SpecularAlbedo", 4),
    "specular_motion_vectors": ("GBuffer.SpecularMvec", 2),
}


class Feature:
    """Feature configuration and the parameter blocks every backend fills in.

    Args:
        input_size: ``(width, height)`` of the noisy images fed to the denoiser.
        output_size: ``(width, height)`` of the denoised images. When it is
            larger than `input_size`, DLSS upscales. Defaults to `input_size`,
            i.e. no upscaling (DLAA mode).
        quality: Quality/performance tradeoff of the upscaling step, one of
            ``"high"``, ``"balanced"`` or ``"fast"``, or a raw
            NVSDK_NGX_PerfQuality_Value. Has no effect without upscaling.
        hdr: The color buffer is high dynamic range.
        auto_exposure: Let DLSS estimate the exposure itself.
        depth_inverted: The depth buffer grows towards the camera.
        mv_low_res: Motion vectors are given at the input resolution.
        mv_jittered: Motion vectors include the sub-pixel jitter.
        hw_depth: The depth buffer holds hardware (post-projection) depth rather
            than linear camera-space depth.
        packed_roughness: Roughness rides in ``normals.w`` instead of being a
            buffer of its own.
    """

    def __init__(self, input_size, output_size=None, quality: str | int = "high", *,
                 hdr: bool = True, auto_exposure: bool = False, depth_inverted: bool = False,
                 mv_low_res: bool = True, mv_jittered: bool = False, hw_depth: bool = False,
                 packed_roughness: bool = True):
        # Not `output_size or input_size`: a vector type refuses to be a bool.
        self.input_size = tuple(int(v) for v in input_size)
        self.output_size = tuple(int(v) for v in
                                 (input_size if output_size is None else output_size))
        self.quality = quality
        self.packed_roughness = packed_roughness
        self.hw_depth = hw_depth
        self.last_error: str | None = None
        self._reset = True  # the first evaluation has no history to keep

        if min(self.input_size) <= 0:
            raise DLSSError("the input size must be non-zero")
        if (self.output_size[0] < self.input_size[0]
                or self.output_size[1] < self.input_size[1]):
            raise DLSSError(f"the output size {self.output_size} cannot be smaller than the "
                            f"input size {self.input_size}, DLSS does not downscale")
        if min(self.input_size) < MIN_EXTENT:
            raise DLSSError(f"the input size {self.input_size} is too small, DLSS requires at "
                            f"least {MIN_EXTENT} x {MIN_EXTENT} pixels")
        if not isinstance(quality, int) and quality not in QUALITY:
            raise DLSSError(f'unknown quality mode "{quality}", expected one of '
                            f'"high", "balanced" or "fast"')

        self.flags = 0
        self.flags |= FLAG_IS_HDR if hdr else 0
        self.flags |= FLAG_AUTO_EXPOSURE if auto_exposure else 0
        self.flags |= FLAG_DEPTH_INVERTED if depth_inverted else 0
        self.flags |= FLAG_MV_LOW_RES if mv_low_res else 0
        self.flags |= FLAG_MV_JITTERED if mv_jittered else 0

    @property
    def upscale_factor(self) -> float:
        return self.output_size[0] / self.input_size[0]

    def perf_quality(self) -> int:
        if isinstance(self.quality, int):
            return self.quality
        if self.upscale_factor == 1.0:
            return PERF_DLAA
        if self.quality == "fast" and self.upscale_factor >= 3.0:
            return PERF_ULTRA_PERFORMANCE
        return QUALITY[self.quality]

    def fill_create_params(self, params: Parameters):
        params.set_uint("CreationNodeMask", 1)
        params.set_uint("VisibilityNodeMask", 1)
        params.set_uint("Width", self.input_size[0])
        params.set_uint("Height", self.input_size[1])
        params.set_uint("OutWidth", self.output_size[0])
        params.set_uint("OutHeight", self.output_size[1])
        params.set_int("PerfQualityValue", self.perf_quality())
        params.set_int("DLSS.Denoise.Mode", 1)  # DLUnified
        params.set_int("DLSS.Feature.Create.Flags", self.flags)
        params.set_int("DLSS.Enable.Output.Subrects", 0)
        params.set_uint("DLSS.Use.HW.Depth", 1 if self.hw_depth else 0)
        params.set_uint("DLSS.Roughness.Mode", 1 if self.packed_roughness else 0)

    def fill_eval_params(self, params: Parameters, jitter, mv_scale, reset: bool,
                         pre_exposure: float, exposure_scale: float):
        params.set_int("Reset", 1 if reset else 0)
        params.set_float("Jitter.Offset.X", float(jitter[0]))
        params.set_float("Jitter.Offset.Y", float(jitter[1]))
        params.set_float("MV.Scale.X", float(mv_scale[0]))
        params.set_float("MV.Scale.Y", float(mv_scale[1]))
        params.set_uint("DLSS.Render.Subrect.Dimensions.Width", self.input_size[0])
        params.set_uint("DLSS.Render.Subrect.Dimensions.Height", self.input_size[1])
        params.set_float("DLSS.Pre.Exposure", pre_exposure)
        params.set_float("DLSS.Exposure.Scale", exposure_scale)
        # Mitsuba images are stored top to bottom, like the textures DLSS wants.
        params.set_int("DLSS.Indicator.Invert.X.Axis", 0)
        params.set_int("DLSS.Indicator.Invert.Y.Axis", 0)

    def input_names(self, given: dict, extra: dict | None, resolve) -> dict:
        """Name the buffers the way NGX does, dropping the ones not supplied.

        `resolve` turns one caller-supplied buffer into whatever the backend
        binds, given its NGX name and channel count -- None for the buffers
        passed through `extra`, whose layout only the caller knows.
        """
        if given.get("roughness") is None and not self.packed_roughness:
            raise DLSSError("pass roughness, or construct with packed_roughness=True")
        if given.get("roughness") is not None and self.packed_roughness:
            raise DLSSError("packed_roughness=True expects roughness in normals.w, so a "
                            "separate roughness buffer would be ignored")
        inputs = {}
        for keyword, (name, channels) in INPUTS.items():
            if (value := given.get(keyword)) is not None:
                inputs[name] = resolve(value, name, channels)
        if self.packed_roughness and "GBuffer.Normals" in inputs:
            # Roughness rides in normals.w, and NGX wants both bound to it.
            inputs["GBuffer.Roughness"] = inputs["GBuffer.Normals"]
        for name, value in (extra or {}).items():
            inputs[name] = resolve(value, name, None)
        return inputs

    def __repr__(self):
        return (f"{type(self).__name__}[\n  input_size = {self.input_size},\n"
                f'  output_size = {self.output_size},\n  quality = "{self.quality}"\n]')

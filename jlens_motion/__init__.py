"""Jacobian lens (J-lens) for the MotionMillion text-to-motion LLaMA.

See ``jlens_motion/README.md`` and ``docs/07_jlens_verification_plan.md``.
"""

from jlens_motion.fitting import (
    POSITION_MODES,
    fit,
    fit_offsets,
    jacobian_for_example,
    motion_to_motion,
    offset_jacobians_for_example,
    text_to_motion,
)
from jlens_motion.hooks import ActivationRecorder
from jlens_motion.lens import JacobianLens, OffsetJacobians, lens_readout
from jlens_motion.model import (
    MotionExample,
    MotionLensModel,
    motion_positions,
    text_positions,
)

__all__ = [
    "POSITION_MODES",
    "ActivationRecorder",
    "JacobianLens",
    "MotionExample",
    "MotionLensModel",
    "OffsetJacobians",
    "fit",
    "fit_offsets",
    "jacobian_for_example",
    "lens_readout",
    "motion_positions",
    "motion_to_motion",
    "offset_jacobians_for_example",
    "text_positions",
    "text_to_motion",
]

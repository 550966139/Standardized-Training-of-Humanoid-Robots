"""Input format adapters."""
from .base import DataSource, MotionData, SkeletonSchema
from .registry import detect_format, load

__all__ = ["DataSource", "MotionData", "SkeletonSchema", "detect_format", "load"]

"""Counter-drone detection and tracking with co-located RGB + thermal cameras."""

from .config import Config
from .pipeline import CounterDronePipeline, FrameResult

__all__ = ["Config", "CounterDronePipeline", "FrameResult"]
__version__ = "0.1.0"

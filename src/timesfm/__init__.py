"""TimesFM: A time series forecasting model.

This package provides a PyTorch-based implementation of the TimesFM
foundation model for time series forecasting.

Note: Personal fork for learning/experimentation. Upstream: google-research/timesfm
"""

from timesfm.model import TimesFM
from timesfm.config import TimesFMConfig

__version__ = "0.1.0"
__all__ = ["TimesFM", "TimesFMConfig"]

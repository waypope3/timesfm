"""TimesFM: A time series forecasting model.

This package provides a PyTorch-based implementation of the TimesFM
foundation model for time series forecasting.

Note: Personal fork for learning/experimentation. Upstream: google-research/timesfm

Personal notes:
- Experimenting with custom context lengths and patch sizes
- See experiments/ directory for notebooks
"""

from timesfm.model import TimesFM
from timesfm.config import TimesFMConfig

__version__ = "0.1.0"
__author_fork__ = "personal learning fork"
__all__ = ["TimesFM", "TimesFMConfig"]

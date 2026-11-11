"""Adaptive drift-aware phishing URL detection framework.

Re-implementation of Krishnan & Naik (Security and Privacy, 2025,
DOI 10.1002/spy2.70057): lexical URL features, LASSO + mutual-information
feature selection, Kolmogorov-Smirnov drift monitoring and online SGD.
"""

from .framework import DriftAwarePhishingDetector, FrameworkConfig

__all__ = ["DriftAwarePhishingDetector", "FrameworkConfig"]
__version__ = "2.0.0"

"""Segmentation subpackage placeholder for packaging.

Makes the `segmentation/` directory a proper Python package so that configs and helper
modules (e.g., `model.py`) are installable. Downstream projects can then import
`vmamba.segmentation.model` if needed.
"""

from . import model as _model  # noqa: F401

__all__ = [
    "_model",
]

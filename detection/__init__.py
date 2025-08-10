"""Detection subpackage placeholder for packaging.

Currently this project ships training scripts & configs inside `detection/`.
Adding this file allows setuptools to discover and install the directory as a package.

Public surface kept intentionally slim; users typically work through configs / external frameworks.
"""

from . import model as _model  # noqa: F401  (expose for introspection)

__all__ = [
    "_model",
]

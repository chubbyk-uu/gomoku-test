"""Build configuration for optional Cython extensions."""

from pathlib import Path

import numpy
from setuptools import Extension, setup


def build_extensions() -> list[Extension]:
    """Return the list of native extensions to build."""
    source_root = Path("src/gomoku/ai")
    sources = [str(source_root / "_threat_kernels.pyx")]

    extension = Extension(
        "gomoku.ai._threat_kernels",
        sources=sources,
        include_dirs=[numpy.get_include()],
    )

    try:
        from Cython.Build import cythonize
    except ImportError:
        return [extension]

    return cythonize(
        [extension],
        compiler_directives={"language_level": "3"},
    )


setup(ext_modules=build_extensions())

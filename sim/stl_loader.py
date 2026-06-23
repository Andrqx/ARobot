"""Minimal STL mesh loader — pure Python + numpy, no extra dependency.

Reads both flavours SolidWorks can export:
  * **binary STL** (the default, compact), and
  * **ASCII STL** (text, ``solid ... facet normal ...``).

Returns the triangles as a single ``(n_triangles, 3, 3)`` float array — for
each triangle, its three ``(x, y, z)`` vertices — which is exactly what
matplotlib's ``Poly3DCollection`` wants. Units are whatever the file is in
(export in millimetres to match the rest of the stack).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# Binary STL layout: 80-byte header, uint32 triangle count, then per triangle
# 12 little-endian float32s (a normal + 3 vertices) plus a 2-byte attribute.
_BINARY_HEADER = 84          # 80 header + 4 count
_BINARY_TRIANGLE = 50        # 12*4 floats + 2 attribute bytes
_BINARY_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("v", "<f4", (3, 3)),
    ("attr", "<u2"),
])


def load_stl(path: str | Path) -> np.ndarray:
    """Load an STL file -> ``(n, 3, 3)`` array of triangle vertices."""
    data = Path(path).read_bytes()
    if _looks_binary(data):
        return _parse_binary(data)
    return _parse_ascii(data.decode("ascii", errors="replace"))


def _looks_binary(data: bytes) -> bool:
    """True if the byte length matches the binary triangle-count exactly.

    This size check is the reliable test — an ASCII file that merely starts
    with the word ``solid`` won't satisfy it, and a binary file whose header
    happens to spell ``solid`` still will.
    """
    if len(data) < _BINARY_HEADER:
        return False
    (n,) = struct.unpack_from("<I", data, 80)
    return len(data) == _BINARY_HEADER + n * _BINARY_TRIANGLE


def _parse_binary(data: bytes) -> np.ndarray:
    (n,) = struct.unpack_from("<I", data, 80)
    if n == 0:
        return np.empty((0, 3, 3), dtype=float)
    rec = np.frombuffer(data, dtype=_BINARY_DTYPE, count=n, offset=_BINARY_HEADER)
    return rec["v"].astype(np.float64)


def _parse_ascii(text: str) -> np.ndarray:
    verts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            _, x, y, z = line.split()[:4]
            verts.append((float(x), float(y), float(z)))
    arr = np.asarray(verts, dtype=float)
    if arr.size == 0 or arr.shape[0] % 3 != 0:
        raise ValueError(f"malformed ASCII STL: got {arr.shape[0]} vertices "
                         "(expected a multiple of 3)")
    return arr.reshape(-1, 3, 3)

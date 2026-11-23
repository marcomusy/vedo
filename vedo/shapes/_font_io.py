from __future__ import annotations

"""Versioned, pickle-free NPZ storage for polygonal 3D fonts."""

import numpy as np

from vedo.file_io.numpy_io import load_numpy


def _validate_glyph(points, faces):
    if (not isinstance(points, np.ndarray) or points.ndim != 2
            or points.shape[1] not in (2, 3) or points.dtype.kind not in "fiu"
            or not np.isfinite(points).all()):
        raise ValueError("Font points must be a finite numeric array of shape (n, 2 or 3)")
    if (not isinstance(faces, np.ndarray) or faces.ndim != 2
            or faces.shape[1] != 3 or faces.dtype.kind not in "iu"):
        raise ValueError("Font faces must be an integer array of triangles")
    if faces.size and (faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError("Font face index is outside its glyph's points")


def _validate_offsets(offsets, count, size):
    if (offsets.ndim != 1 or offsets.dtype.kind not in "iu"
            or len(offsets) != count + 1 or offsets[0] != 0
            or offsets[-1] != size or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("Invalid font glyph offsets")


def read_font(filename, *, expected_sha256=None):
    """Read a safe font, or a legacy font subject to the shared pickle policy."""
    data = load_numpy(filename, expected_sha256=expected_sha256)
    if not isinstance(data, dict):
        raise ValueError("Expected a font NPZ archive")

    if "font" in data:
        # Only reachable after explicit opt-in or official checksum verification.
        font = data["font"][0]
        if not isinstance(font, dict):
            raise ValueError("Invalid legacy font dictionary")
        for character, (points, faces) in font.items():
            if not isinstance(character, str) or len(character) != 1:
                raise ValueError("Invalid font character")
            _validate_glyph(points, faces)
        return font

    version = data["font_format"]
    if version.shape != () or version.dtype.kind not in "iu" or version.item() != 1:
        raise ValueError("Unsupported 3D font format version")
    codepoints = data["codepoints"]
    if (codepoints.ndim != 1 or not len(codepoints) or codepoints.dtype.kind not in "iu"
            or np.any(codepoints < 0) or np.any(codepoints > 0x10FFFF)
            or len(set(codepoints)) != len(codepoints)):
        raise ValueError("Font codepoints must be unique Unicode character codes")
    points, faces = data["points"], data["faces"]
    if points.ndim != 2 or faces.ndim != 2:
        raise ValueError("Invalid font geometry dimensions")
    point_offsets, face_offsets = data["point_offsets"], data["face_offsets"]
    _validate_offsets(point_offsets, len(codepoints), len(points))
    _validate_offsets(face_offsets, len(codepoints), len(faces))

    font = {}
    for i, codepoint in enumerate(codepoints):
        glyph_points = points[point_offsets[i]:point_offsets[i + 1]]
        glyph_faces = faces[face_offsets[i]:face_offsets[i + 1]]
        _validate_glyph(glyph_points, glyph_faces)
        font[chr(int(codepoint))] = (glyph_points, glyph_faces)
    return font


def write_font(filename, font):
    """Write numeric arrays and Unicode codepoints, preserving glyph geometry."""
    if not font:
        raise ValueError("Cannot write an empty font")
    for character, (points, faces) in font.items():
        if not isinstance(character, str) or len(character) != 1:
            raise ValueError("Invalid font character")
        _validate_glyph(points, faces)
    points, faces = zip(*font.values())
    # Offsets delimit glyphs; triangle indices remain local to each glyph.
    arrays = dict(
        font_format=np.array(1, dtype=np.uint8),
        # Integer codepoints also preserve the NUL glyph, which NumPy Unicode
        # arrays would turn into an empty string when read back.
        codepoints=np.array([ord(c) for c in font], dtype=np.uint32),
        points=np.concatenate(points),
        faces=np.concatenate(faces),
        point_offsets=np.concatenate(([0], np.cumsum([len(p) for p in points], dtype=np.int64))),
        face_offsets=np.concatenate(([0], np.cumsum([len(f) for f in faces], dtype=np.int64))),
    )
    np.savez_compressed(filename, **arrays)

"""Convert a trusted legacy 3D font to the pickle-free font format.

Run from the checkout, for example:
    VEDO_ALLOW_UNSAFE_PICKLE=1 PYTHONPATH=. python tools/migrate_font_archive.py \
        old-font.npz new-font.npz

Legacy input is executable pickle data. Only convert files you already trust.
Filenames do not need to change; use the same path to replace a trusted archive.
"""

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from vedo.shapes._font_io import read_font, write_font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    original = read_font(args.source)
    destination = Path(args.destination)
    # Verify before replacing the destination, including in-place conversions.
    with NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        write_font(temporary_path, original)
        converted = read_font(temporary_path)
        if list(original) != list(converted):
            raise ValueError("Conversion changed the font's character order")
        for character, geometry in original.items():
            for old, new in zip(geometry, converted[character]):
                if old.dtype != new.dtype:
                    raise ValueError("Conversion changed a glyph's numeric type")
                np.testing.assert_array_equal(old, new)
        mode = destination.stat().st_mode & 0o777 if destination.exists() else 0o644
        temporary_path.chmod(mode)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Converted and verified {len(original)} glyphs: {args.destination}")


if __name__ == "__main__":
    main()

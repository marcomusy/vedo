# vedo.file_io

## NumPy files and trusted legacy data

Pickle loading is disabled by default. Legacy vedo `.npy` assemblies and
`.npy`/`.npz` scenes contain pickled Python objects, which can execute arbitrary
code while being read, even if loading later fails. This policy also applies to
URL, directory, compressed-file, constructor, and CLI loading paths.

For files you already trust, enable the compatibility option explicitly:

```bash
VEDO_ALLOW_UNSAFE_PICKLE=1 python my_script.py
VEDO_ALLOW_UNSAFE_PICKLE=1 vedo trusted-scene.npz
```

Only the exact value `1` enables this option; unset, `0`, and other values keep
pickle disabled. The option affects the entire process, including downloaded
files. Vedo emits a warning once when the option is first used to unpickle data.
There is no automatic unsafe retry without explicit opt-in.

Scene and assembly writers still produce their existing legacy format, so their
output also requires this option when read back. Numeric NumPy arrays do not
require pickle; the `.npy` and `.npz` extensions alone do not imply pickle.

Bundled 3D fonts have been converted to numeric arrays and need no opt-in.
Official downloaded fonts keep their existing filenames and contents for older
vedo releases. New releases verify their SHA-256 checksums against values shipped
inside the package before permitting legacy deserialization. Verification uses
the same bytes that are subsequently loaded; a mismatch is rejected even when
the compatibility option is enabled. Cached fonts can be verified offline.
Unrecognized custom legacy fonts require explicit opt-in.

To convert a trusted custom font from a source checkout, use:

```bash
VEDO_ALLOW_UNSAFE_PICKLE=1 PYTHONPATH=. python tools/migrate_font_archive.py old-font.npz new-font.npz
```

The tool checks that glyph order, numeric types, coordinates and triangle indices
are preserved. The new font then loads without the compatibility option. The
new format stores a version number, Unicode codepoints, point and triangle
arrays, and offsets delimiting each glyph; it contains no object arrays.

## API

::: vedo.file_io.loaders
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false

::: vedo.file_io.network
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false

::: vedo.file_io.scene
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false

::: vedo.file_io.terminal
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false

::: vedo.file_io.video
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false

::: vedo.file_io.writers
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false

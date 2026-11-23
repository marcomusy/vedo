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

New scene exports, including the `E` shortcut and `export_window()`, use a
versioned pickle-free format and load without the compatibility option. NPZ
scenes store JSON metadata beside typed NumPy arrays; NPY scenes store UTF-8 JSON
with embedded pickle-free arrays. NPZ is preferred for compact storage of larger
scenes. Geometry, array types, camera state and ordinary metadata are preserved.
Object arrays and arbitrary Python classes cannot be exported; convert custom
metadata to plain containers or typed arrays first. A failed export leaves an
existing output file intact.

Mesh geometry is checked before reconstruction: coordinates must be finite real
arrays with two or three components, and polygon/line connectivity must contain
complete cells with integer vertex indices inside the coordinate array. These
checks also apply to legacy NumPy geometry and nested assembly parts. Invalid
geometry is rejected before being passed to VTK.

To convert an old trusted scene, open it once with the compatibility option and
press `E` to save a new `scene.npz`, or use `export_window()` from Python. The new
file can then be opened with the option unset. Older vedo releases do not support
the new scene format.

Assembly writers still produce their existing legacy format, so assembly files
require the compatibility option when read back. Numeric NumPy arrays do not
require pickle; the `.npy` and `.npz` extensions alone do not imply pickle.

Bundled 3D fonts have been converted to numeric arrays and need no opt-in.
Official downloaded fonts keep their existing filenames and contents for older
vedo releases. New releases verify their SHA-256 checksums against values shipped
inside the package before permitting legacy deserialization. Verification uses
the same bytes that are subsequently loaded; a mismatch is rejected even when
the compatibility option is enabled. Cached fonts can be verified offline.
Unrecognized custom legacy fonts require explicit opt-in.

The numeric font format stores a version number, Unicode codepoints, point and
triangle arrays, and offsets delimiting each glyph. It contains no object arrays
and loads without the compatibility option.

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

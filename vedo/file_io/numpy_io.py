from __future__ import annotations

"""Shared policy for loading NumPy files without implicit pickle execution."""

import hashlib
import io
import os
import warnings
from functools import lru_cache

import numpy as np


class UnsafePickleError(ValueError):
    """A NumPy file requires pickle, but the caller has not trusted it."""


class NumpyIntegrityError(ValueError):
    """An official font does not match its packaged checksum."""


class UnsafePickleWarning(RuntimeWarning):
    """The explicit compatibility option is permitting pickle execution."""


@lru_cache(maxsize=1)
def _warn_unsafe_pickle() -> None:
    warnings.warn(
        "VEDO_ALLOW_UNSAFE_PICKLE=1 permits legacy NumPy files to execute Python "
        "code. Only load files you trust. This setting affects the whole process.",
        UnsafePickleWarning,
        stacklevel=3,
    )


def _requires_pickle(error: ValueError) -> bool:
    # NumPy's wording differs across supported releases. Keep malformed numeric
    # data/header errors distinct from the errors that disable object loading.
    message = str(error)
    return any(text in message for text in (
        "Object arrays cannot be loaded when allow_pickle=False",
        "Cannot load file containing pickled data when allow_pickle=False",
        "This file contains pickled (object) data",
    ))


def _read_numpy(stream, *, key, encoding, verified):
    def read(allow_pickle):
        stream.seek(0)
        result = np.load(stream, allow_pickle=allow_pickle, encoding=encoding)
        if isinstance(result, np.lib.npyio.NpzFile):
            with result:
                if key is not None:
                    return result[key]
                return {name: result[name] for name in result.files}
        if key is not None:
            raise ValueError("Expected a NumPy NPZ archive")
        return result

    try:
        return read(False)
    except ValueError as error:
        if not _requires_pickle(error):
            raise
        if not verified:
            if os.environ.get("VEDO_ALLOW_UNSAFE_PICKLE") != "1":
                raise UnsafePickleError(
                    "Loading pickled NumPy data is disabled. Legacy vedo scenes, "
                    "assemblies and custom fonts can execute arbitrary Python code. "
                    "Only for files you trust, set VEDO_ALLOW_UNSAFE_PICKLE=1 "
                    "before loading them."
                ) from error
            _warn_unsafe_pickle()
        return read(True)


def load_numpy(filename, *, key=None, encoding="ASCII", expected_sha256=None):
    """Load an array or archive, allowing pickle only by opt-in or verified hash.

    NPZ members are read before closing the archive. For an official legacy font,
    verify and deserialize the same byte snapshot to avoid a file replacement
    between the integrity check and unpickling. Hashes must come from the package,
    never from the downloaded file or its server.
    """
    with open(filename, "rb") as source:
        if expected_sha256 is not None:
            content = source.read()
            if hashlib.sha256(content).hexdigest() != expected_sha256:
                raise NumpyIntegrityError(
                    f"Font checksum mismatch for {os.fspath(filename)!r}; "
                    "refusing to load an unverified official font."
                )
            with io.BytesIO(content) as snapshot:
                return _read_numpy(snapshot, key=key, encoding=encoding, verified=True)
        return _read_numpy(source, key=key, encoding=encoding, verified=False)

from __future__ import annotations

from importlib import import_module


def build_attr_map(*groups):
    """Build an ordered lazy export map from `(module_name, exports)` groups."""
    attr_map = {}
    for module_name, exports in groups:
        for export in exports:
            if isinstance(export, tuple):
                public_name, target_name = export
            else:
                public_name = target_name = export
            attr_map[public_name] = (module_name, target_name)
    return attr_map, list(attr_map)


def getattr_lazy(module_name, module_globals, name, attr_map=None, module_map=None):
    """Resolve a lazily-exported attribute or submodule and cache it."""
    if attr_map is not None and name in attr_map:
        target_module, target_name = attr_map[name]
        try:
            value = getattr(import_module(target_module), target_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"lazy import of {name!r} from {target_module!r} failed"
            ) from e
        module_globals[name] = value
        return value

    if module_map is not None and name in module_map:
        value = import_module(module_map[name])
        module_globals[name] = value
        return value

    raise AttributeError(f"module {module_name!r} has no attribute {name!r}")


def dir_lazy(module_globals, attr_map=None, module_map=None):
    """Return `dir()` names including lazy exports and lazy submodules."""
    names = set(module_globals)
    if attr_map is not None:
        names.update(attr_map)
    if module_map is not None:
        names.update(module_map)
    return sorted(names)

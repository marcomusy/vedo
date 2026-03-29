from __future__ import annotations
"""Three.js scene export helpers."""

import base64
import json

import numpy as np

from vedo import colors

from .constants import _threejs_html_template

__docformat__ = "google"


def _pack_numeric_array(value, dtype, min_values=64):
    """Pack large numeric arrays as base64-encoded typed buffers."""
    if value is None:
        return []
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        return []
    flat = np.ascontiguousarray(arr.reshape(-1))
    if flat.size < min_values:
        return flat.tolist()
    return {
        "encoding": "base64",
        "dtype": np.dtype(dtype).name,
        "length": int(flat.size),
        "data": base64.b64encode(flat.tobytes()).decode("ascii"),
    }


def _flatten_numeric_array(value) -> list[float]:
    """Flatten an array-like value to a 1D Python list."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return arr.reshape(-1).tolist()


def _triangulate_flat_cells(flat_cells) -> list[int]:
    """Convert VTK-style polygon connectivity into triangle indices."""
    if flat_cells is None:
        return []
    cells = np.asarray(flat_cells, dtype=int).ravel()
    triangles = []
    i = 0
    ncells = len(cells)
    while i < ncells:
        nids = int(cells[i])
        ids = cells[i + 1 : i + 1 + nids]
        if nids >= 3:
            root = int(ids[0])
            for j in range(1, nids - 1):
                triangles.extend([root, int(ids[j]), int(ids[j + 1])])
        i += nids + 1
    return triangles


def _polygon_edge_segments(flat_cells, points) -> list[float]:
    """Build unique polygon edge segments without triangulation diagonals."""
    if flat_cells is None:
        return []
    cells = np.asarray(flat_cells, dtype=int).ravel()
    pts = np.asarray(points, dtype=float)
    segments = []
    seen = set()
    i = 0
    ncells = len(cells)
    while i < ncells:
        nids = int(cells[i])
        ids = cells[i + 1 : i + 1 + nids]
        if nids >= 2:
            for j in range(nids):
                a = int(ids[j])
                b = int(ids[(j + 1) % nids])
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                segments.extend(pts[[a, b]].reshape(-1).tolist())
        i += nids + 1
    return segments


def _map_scalars_to_colors(values, lut, value_range) -> list[float]:
    """Map scalar values to vertex colors using a lookup table."""
    if values is None or lut is None:
        return []

    arr = np.asarray(values, dtype=float).ravel()
    table = np.asarray(lut, dtype=float)
    if arr.size == 0 or table.size == 0:
        return []

    if value_range is None:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    else:
        lo, hi = map(float, value_range)

    if hi <= lo:
        idx = np.zeros(arr.shape[0], dtype=int)
    else:
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        idx = np.rint(scaled * (len(table) - 1)).astype(int)
    return table[idx, :3].astype(float).reshape(-1).tolist()


def _extract_vertex_colors(obj: dict, npoints: int) -> list[float]:
    """Extract point-based colors when available."""
    pointdata = obj.get("pointdata") or {}
    candidate_names = []
    array_name = obj.get("array_name_to_color_by")
    if array_name:
        candidate_names.append(array_name)
    candidate_names.extend(["RGBA", "VertexColors", "Colors", "Scalars"])

    for name in candidate_names:
        if name not in pointdata:
            continue
        arr = np.asarray(pointdata[name])
        if arr.ndim == 2 and arr.shape[0] == npoints and arr.shape[1] in (3, 4):
            cols = arr[:, :3].astype(float)
            if cols.size and np.nanmax(cols) > 1.0:
                cols /= 255.0
            return cols.reshape(-1).tolist()
        if (
            arr.ndim == 1
            and arr.shape[0] == npoints
            and obj.get("scalar_visibility")
            and obj.get("LUT") is not None
        ):
            return _map_scalars_to_colors(arr, obj.get("LUT"), obj.get("scalar_range"))
    return []


def _extract_texture_payload(obj: dict, pack_arrays=True) -> tuple[dict | None, dict | None]:
    """Extract mesh texture data when present and supported."""
    texture_array = obj.get("texture_array")
    if texture_array is None:
        return None, None

    uvs = np.asarray(obj.get("texture_coordinates")) if obj.get("texture_coordinates") is not None else None
    if uvs is None or uvs.size == 0:
        label = obj.get("name") or obj.get("filename") or "Mesh"
        return None, {"label": f"{label} (texture without UVs)"}

    image = np.asarray(texture_array)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        label = obj.get("name") or obj.get("filename") or "Mesh"
        return None, {"label": f"{label} (unsupported texture format)"}

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = image.astype(np.uint8)

    return (
        {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "channels": int(image.shape[2]),
            "data": _pack_numeric_array(image, np.uint8, min_values=256) if pack_arrays else image.reshape(-1).tolist(),
            "repeat": bool(obj.get("texture_repeat")),
            "interpolate": bool(obj.get("texture_interpolate")),
        },
        None,
    )


def _scene_object_to_threejs(obj: dict, pack_arrays=True) -> tuple[dict | None, dict | None]:
    """Convert a serialized scene object to a Three.js-friendly payload."""
    from .scene import _color_to_hex

    otype = obj.get("type", "unknown")
    label = obj.get("name") or obj.get("filename") or otype

    if otype == "Text2D":
        bgcol = obj.get("bgcol")
        alpha_value = obj.get("alpha")
        alpha = 0.0 if alpha_value is None else float(alpha_value)
        background = None
        if bgcol is not None and alpha > 0:
            r, g, b = colors.get_color(bgcol)
            background = "rgba({:.0f}, {:.0f}, {:.0f}, {:.3f})".format(
                r * 255, g * 255, b * 255, alpha
            )
        return (
            {
                "kind": "text2d",
                "name": label,
                "text": obj.get("text", ""),
                "position": [float(obj["position"][0]), float(obj["position"][1])],
                "color": _color_to_hex(obj.get("color")),
                "size": float(1.0 if obj.get("size") is None else obj.get("size")),
                "frame": bool(obj.get("frame")),
                "background": background,
            },
            None,
        )

    if otype != "Mesh":
        return None, {"label": f"{label} ({otype})"}

    points = np.asarray(obj.get("points"), dtype=float)
    if points.size == 0:
        return None, None

    base = {
        "name": label,
        "color": _color_to_hex(obj.get("color")) or "#d9dde3",
        "opacity": float(1.0 if obj.get("alpha") is None else obj.get("alpha")),
        "flat_shading": int(obj.get("shading", 1)) == 0,
        "wireframe": int(obj.get("representation", 0)) == 1,
        "edge_visibility": bool(obj.get("edge_visibility")),
        "edge_color": _color_to_hex(obj.get("linecolor")) or _color_to_hex(obj.get("color")) or "#d9dde3",
        "line_width": float(1.0 if obj.get("linewidth") is None else obj.get("linewidth")),
        "ambient": float(0.0 if obj.get("ambient") is None else obj.get("ambient")),
        "diffuse": float(1.0 if obj.get("diffuse") is None else obj.get("diffuse")),
        "specular": float(0.0 if obj.get("specular") is None else obj.get("specular")),
        "specular_power": float(1.0 if obj.get("specularpower") is None else obj.get("specularpower")),
        "specular_color": _color_to_hex(obj.get("specularcolor")) or "#111111",
        "lighting": bool(True if obj.get("lighting_is_on") is None else obj.get("lighting_is_on")),
        "back_color": _color_to_hex(obj.get("backcolor")),
    }
    vertex_colors = _extract_vertex_colors(obj, len(points))
    texture_payload, texture_warning = _extract_texture_payload(obj, pack_arrays=pack_arrays)

    cells = obj.get("cells")
    if cells is not None and len(np.asarray(cells).ravel()) > 0:
        return (
            {
                **base,
                "kind": "mesh",
                "positions": _pack_numeric_array(points, np.float32) if pack_arrays else points.reshape(-1).tolist(),
                "indices": (
                    _pack_numeric_array(
                        _triangulate_flat_cells(cells),
                        np.uint16 if len(points) < 65536 else np.uint32,
                    )
                    if pack_arrays else _triangulate_flat_cells(cells)
                ),
                "normals": (
                    _pack_numeric_array(obj.get("point_normals"), np.float32)
                    if pack_arrays else _flatten_numeric_array(obj.get("point_normals"))
                ),
                "uvs": (
                    _pack_numeric_array(obj.get("texture_coordinates"), np.float32)
                    if pack_arrays else _flatten_numeric_array(obj.get("texture_coordinates"))
                ),
                "vertex_colors": (
                    _pack_numeric_array(vertex_colors, np.float32)
                    if pack_arrays else vertex_colors
                ),
                "texture": texture_payload,
                "edge_segments": (
                    _pack_numeric_array(_polygon_edge_segments(cells, points), np.float32)
                    if pack_arrays else _polygon_edge_segments(cells, points)
                ),
            },
            texture_warning,
        )

    lines = obj.get("lines")
    if lines is not None and len(lines):
        polylines = []
        for line in lines:
            ids = np.asarray(line, dtype=int).ravel()
            if ids.size < 2:
                continue
            polylines.append(points[ids].reshape(-1).tolist())
        return (
            {
                **base,
                "kind": "lines",
                "polylines": polylines,
                "vertex_colors": _pack_numeric_array(vertex_colors, np.float32) if pack_arrays else vertex_colors,
            },
            None,
        )

    return (
        {
            **base,
            "kind": "points",
            "positions": _pack_numeric_array(points, np.float32) if pack_arrays else points.reshape(-1).tolist(),
            "point_size": max(float(4.0 if obj.get("pointsize") is None else obj.get("pointsize")), 1.0),
            "vertex_colors": _pack_numeric_array(vertex_colors, np.float32) if pack_arrays else vertex_colors,
        },
        None,
    )


def _normalize_threejs_options(options: dict | None = None) -> dict:
    """Validate and normalize Three.js export tuning options."""
    defaults = {
        "headlight_intensity": 1.0,
        "ambient_scale": 0.25,
        "specular_scale": 0.45,
        "fallback_specular_strength": 0.35,
        "fallback_shininess": 28.0,
        "preserve_base_color": False,
        "pack_arrays": True,
    }
    if not options:
        return defaults

    normalized = dict(defaults)
    for key, value in options.items():
        if key not in normalized:
            continue
        if key in ("preserve_base_color", "pack_arrays"):
            normalized[key] = bool(value)
        else:
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                pass

    normalized["headlight_intensity"] = max(normalized["headlight_intensity"], 0.0)
    normalized["ambient_scale"] = max(normalized["ambient_scale"], 0.0)
    normalized["specular_scale"] = max(normalized["specular_scale"], 0.0)
    normalized["fallback_specular_strength"] = max(normalized["fallback_specular_strength"], 0.0)
    normalized["fallback_shininess"] = max(normalized["fallback_shininess"], 1.0)
    return normalized


def _scene_to_threejs_payload(plt, backend_options: dict | None = None) -> dict:
    """Build a Three.js-specific scene payload from the current Plotter."""
    from .scene import _color_to_hex, _json_compatible, _plotter_to_scene_dict

    scene = _plotter_to_scene_dict(plt)
    has_real_axes = any(
        isinstance(obj.get("metadata"), dict) and obj["metadata"].get("assembly") == "Axes"
        for obj in scene.get("objects", [])
    )
    payload = {
        "title": scene.get("title") or "vedo scene",
        "camera": {
            "position": _json_compatible(scene["camera"]["pos"]),
            "focal_point": _json_compatible(scene["camera"]["focal_point"]),
            "viewup": _json_compatible(scene["camera"]["viewup"]),
            "parallel_projection": bool(scene.get("use_parallel_projection")),
            "parallel_scale": float(
                1.0 if scene["camera"].get("parallel_scale") is None else scene["camera"].get("parallel_scale")
            ),
        },
        "background": {
            "primary": _color_to_hex(scene.get("backgrcol")),
            "secondary": _color_to_hex(scene.get("backgrcol2")),
        },
        "threejs": _normalize_threejs_options(backend_options),
        "helpers": {
            "show_axes": bool(scene.get("axes")) and not has_real_axes,
            "axes_size": 1.0,
            "show_grid": scene.get("axes") in (1, 11) and not has_real_axes,
            "grid_size": 10.0,
            "grid_divisions": 10,
        },
        "objects": [],
        "unsupported": [],
    }
    pack_arrays = bool(payload["threejs"].get("pack_arrays", True))

    for obj in scene.get("objects", []):
        three_obj, unsupported = _scene_object_to_threejs(obj, pack_arrays=pack_arrays)
        if three_obj is not None:
            payload["objects"].append(three_obj)
        if unsupported is not None:
            payload["unsupported"].append(unsupported)
    return payload


def _export_threejs(plt, fileoutput="scene.html", backend_options: dict | None = None) -> None:
    """Export the current scene as a standalone Three.js HTML document."""
    from .scene import _json_compatible

    payload = _scene_to_threejs_payload(plt, backend_options=backend_options)
    primary = payload["background"]["primary"] or "#1f2937"
    secondary = payload["background"]["secondary"]
    if secondary:
        background = f"linear-gradient(180deg, {primary} 0%, {secondary} 100%)"
    else:
        background = primary

    scene_data = json.dumps(_json_compatible(payload), separators=(",", ":"))
    scene_data = scene_data.replace("</script>", "<\\/script>")
    html = _threejs_html_template.replace("~title", payload["title"])
    html = html.replace("~background", background)
    html = html.replace("~scene_data", scene_data)
    with open(fileoutput, "w", encoding="UTF-8") as outF:
        outF.write(html)

"""Mesh3d sphere rendering for atoms.

Plotly `Scatter3d` markers are camera-facing sprites with sizes in pixels —
they don't shrink/grow with distance. To get true perspective rendering
(closer atoms appear larger) we need actual 3D geometry. This module builds
one icosphere template and instantiates it once per atom into a single
`Mesh3d` trace, which is fast enough for ~150 atoms.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _icosphere(subdivisions: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for a unit icosphere subdivided `n` times.

    subdivisions=0 → 12 vertices, 20 faces (raw icosahedron — too faceted)
    subdivisions=1 → 42 vertices, 80 faces (a good speed/quality balance)
    subdivisions=2 → 162 vertices, 320 faces (heavier; ~150-atom slab gets sluggish)
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts[0])  # normalise to unit sphere

    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=int,
    )

    # Subdivision: split each triangle into 4 by inserting midpoints.
    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []
        verts_list: list[np.ndarray] = list(verts)

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in midpoint_cache:
                return midpoint_cache[key]
            mid = (verts_list[a] + verts_list[b]) / 2.0
            mid /= np.linalg.norm(mid)  # project back to sphere
            verts_list.append(mid)
            idx = len(verts_list) - 1
            midpoint_cache[key] = idx
            return idx

        for face in faces:
            v1, v2, v3 = face
            a = midpoint(v1, v2)
            b = midpoint(v2, v3)
            c = midpoint(v3, v1)
            new_faces.extend(
                [[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]]
            )
        verts = np.array(verts_list)
        faces = np.array(new_faces, dtype=int)

    return verts, faces


# Cache the template — same for every figure.
_TEMPLATE_VERTS, _TEMPLATE_FACES = _icosphere(subdivisions=1)
_N_TEMPLATE_VERTS = len(_TEMPLATE_VERTS)


def make_spheres_trace(
    centers: np.ndarray,
    radius: float,
    colors: list[str] | np.ndarray,
    name: str = "atoms",
    opacity: float = 1.0,
    hovertext: list[str] | None = None,
) -> go.Mesh3d:
    """Build one Mesh3d trace containing N spheres at `centers`.

    Each sphere is a translated/scaled copy of the cached icosphere template.
    Vertex colors come from the per-atom `colors` list, and all template
    vertices for the same atom share that color so each sphere is uniformly
    coloured.
    """
    centers = np.asarray(centers, dtype=float)
    n_atoms = len(centers)
    if n_atoms == 0:
        return go.Mesh3d()

    # Vertices: tile template, scale, translate to each centre.
    template = _TEMPLATE_VERTS * radius
    all_verts = np.empty((n_atoms * _N_TEMPLATE_VERTS, 3), dtype=float)
    for i, c in enumerate(centers):
        start = i * _N_TEMPLATE_VERTS
        all_verts[start : start + _N_TEMPLATE_VERTS] = template + c

    # Faces: tile template faces, offset indices per atom.
    n_template_faces = len(_TEMPLATE_FACES)
    all_faces = np.empty((n_atoms * n_template_faces, 3), dtype=int)
    for i in range(n_atoms):
        all_faces[i * n_template_faces : (i + 1) * n_template_faces] = (
            _TEMPLATE_FACES + i * _N_TEMPLATE_VERTS
        )

    # Vertex colors: one color per atom, repeated for all its template verts.
    vert_colors: list[str] = []
    for c in colors:
        vert_colors.extend([str(c)] * _N_TEMPLATE_VERTS)

    # Per-atom hover text (one entry per vertex, all copies of the atom's text).
    if hovertext is not None:
        vert_hover: list[str] = []
        for h in hovertext:
            vert_hover.extend([h] * _N_TEMPLATE_VERTS)
    else:
        vert_hover = []

    return go.Mesh3d(
        x=all_verts[:, 0],
        y=all_verts[:, 1],
        z=all_verts[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        vertexcolor=vert_colors,
        opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.4),
        lightposition=dict(x=100, y=100, z=200),
        name=name,
        text=vert_hover or None,
        hovertemplate="%{text}<extra></extra>" if vert_hover else None,
        hoverinfo="text" if vert_hover else "skip",
        showlegend=False,
    )

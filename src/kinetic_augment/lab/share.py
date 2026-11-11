"""
The shareable offline Lab: precompute a grid of panes and pack them into one zip.

The zip holds the Lab page (`static/`, classic scripts only) and its data as `<script>` files, so
it opens by double-clicking `index.html` (no server, no models, no network). Data layout, under
the single folder `augmentation-lab/`:

- `data/manifest.js`: `window.LAB_DATA = {profile, schema, profileText, clips, base, panes, stops, rerolls}`
- `data/<clip>/<sha1>.js`: `Lab.addBlob("<sha1>", "<base64 of the gzipped pane blob>")`
- `data/<clip>/frames/NNNN.jpg`: the video frames
- `data/load.js`: writes the `<script>` tag of the manifest (replaces the placeholder); the page
  loads each blob script by adding a tag when the blob is first needed (`Lab.loadScript`)

`plan_grid` lists the panes; `build_share` runs them. Meshes are in every pane: decimated once per
build (`decimate_mesh`) and quantised to a few mm, so the grid, not the meshes, is cut to fit the budget.
"""

from __future__ import annotations

import base64
import dataclasses
import gzip
import hashlib
import json
import logging
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from kinetic_augment.augmentations.profile import Profile, load_profile
from kinetic_augment.lab.bundle import SCALE_I16, encode_pane, jsonable
from kinetic_augment.lab.pane import PaneRequest, run_pane, schema_defaults
from kinetic_augment.lab.schema import PROFILES, build_schema

log = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ROOT = "augmentation-lab"
GRIDS = [(7, 3), (5, 3), (5, 2), (3, 2), (3, 0)]  # (stops per knob, rerolls), richest first
FORBIDDEN_SUFFIXES = (".npz", ".pkl", ".task")
MESH_VERTICES = 2000  # decimated body mesh size in a share build
MESH_MM = (
    3.0  # share meshes are int16 in units of this many millimetres (scale_i16 = 1000 / MESH_MM)
)
GZIP_FACTOR = 1.03  # the zip stores the base64 text about 3 % over the gzipped bytes

README = """Augmentation Lab (shared version)

Double-click index.html. It works offline: nothing to install, no internet needed.
Open it in Chrome or Safari. Drag the knob markers; they snap to settings that were computed
ahead of time. For live knobs on your own videos, see "Want live knobs? Set up full mode" at the
bottom of the page.
"""


# ---- the grid --------------------------------------------------------------------------------
def _decimals(step: float) -> int:
    text = repr(step)
    if "e-" in text:
        return int(text.split("e-")[1])
    return 0 if "." not in text else len(text) - text.index(".") - 1


def _round_to(value: float, step: float) -> float:
    """`knobs.js roundTo`: half up (JavaScript's Math.round), then to the step's decimals."""
    if not step:
        return value
    return round(math.floor(value / step + 0.5) * step, _decimals(step))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b


def _range_model(knob: Dict[str, Any]) -> Tuple[str, float, float]:
    """`knobs.js rangeModel`: (type, lo or centre, squared geometric centre)."""
    d, lo_limit = knob["default"], knob["limits"][0]
    if _same(d[0], d[1]):
        return "single", 0.0, 0.0
    if d[0] <= lo_limit:
        return "fixed", d[0], 0.0
    if 0 < d[0] < 1 < d[1]:
        return "log", math.sqrt(d[0] * d[1]), d[0] * d[1]
    return "linear", (d[0] + d[1]) / 2, 0.0


def range_from_marker(knob: Dict[str, Any], value: float) -> List[float]:
    """`knobs.js rangeFromMarker`: the [lo, hi] a range knob's marker at `value` stands for."""
    lo_limit, hi_limit = knob["limits"]
    step = knob["step"]
    v = _clamp(_round_to(value, step), lo_limit, hi_limit)
    kind, centre, c2 = _range_model(knob)
    if kind == "single":
        return [v, v]
    if kind == "fixed":
        return [centre, max(v, centre)]

    def mirror(x: float) -> float:
        return c2 / x if kind == "log" else 2 * centre - x

    if v >= centre:
        return [max(lo_limit, _round_to(mirror(v), step)), v]
    return [v, min(hi_limit, _round_to(mirror(v), step))]


def knob_stops(knob: Dict[str, Any], stops: int) -> List[Any]:
    """
    The stops of one `share: true` knob, sorted: numbers for a scalar, `[lo, hi]` for a range.

    `stops` values spread evenly over the knob's limits (a range knob's marker is its high end, so
    they spread over the markers that give distinct ranges: from the centre of the default range,
    or from the lower limit when the low end is fixed, up to the upper limit, each turned into its
    range by the same rule the page uses). The default and the proven range's edges inside the
    limits are always included, then duplicates are dropped.
    """
    lo_limit, hi_limit = knob["limits"]
    step = knob["step"]
    proven = knob.get("proven")
    if knob["kind"] == "scalar":
        values = [_round_to(v, step) for v in np.linspace(lo_limit, hi_limit, stops)]
        values.append(knob["default"])
        values.extend(_round_to(p, step) for p in (proven or []) if lo_limit <= p <= hi_limit)
        if knob.get("integer"):
            values = [int(round(v)) for v in values]
        out: List[Any] = []
        for v in sorted(values):
            if not any(_same(v, o) for o in out):
                out.append(v)
        return out
    kind, centre, _ = _range_model(knob)
    start = lo_limit if kind in ("single", "fixed") else max(lo_limit, centre)
    out = [range_from_marker(knob, m) for m in np.linspace(start, hi_limit, stops)]
    out.append(list(knob["default"]))
    for p in proven or []:
        if lo_limit <= p <= hi_limit:
            out.append(range_from_marker(knob, p))
    out.sort(key=lambda r: (r[1], r[0]))
    unique: List[Any] = []
    for r in out:
        if not any(_same(r, o) for o in unique):
            unique.append(r)
    return unique


def stops_table(schema: Dict[str, Any], stops: int) -> Dict[str, Dict[str, List[Any]]]:
    """`{entry id: {knob key: stops}}` for every `share: true` knob (what `LAB_DATA.stops` holds)."""
    table: Dict[str, Dict[str, List[Any]]] = {}
    for entry in schema["entries"]:
        knobs = {
            k["key"]: knob_stops(k, stops)
            for k in entry["knobs"]
            if k.get("share") and k["kind"] in ("scalar", "range")
        }
        if knobs:
            table[entry["id"]] = knobs
    return table


def _enum_values(knob: Dict[str, Any]) -> List[Any]:
    """Each option of an enum knob as a knob value (a one-item list for a multi-choice knob)."""
    return [[o] if knob.get("multi") else o for o in knob.get("options", [])]


def plan_grid(schema: Dict[str, Any], stops: int, rerolls: int, profile: str = "default"):
    """
    Every pane to precompute, as `PaneRequest`s with no clip id (`clip_id=""`), deduplicated.

    Per entry: each stop of each `share: true` knob in random draw 0 and, when the entry has an
    extreme switch, extreme low and high (the stop at the default is the default pane, shared by
    all knobs); random draws 1..`rerolls` at the defaults; each enum option at the defaults. Plus
    the whole profiles `combined.default`, `combined.ambitious`, `combined.literature` (draw 0 and
    the rerolls). Requests are unique by `PaneRequest.share_key`.
    """
    seen: Dict[str, PaneRequest] = {}

    def add(req: PaneRequest, defaults: Dict[str, Any]) -> None:
        seen.setdefault(req.share_key(defaults), req)

    for entry in schema["entries"]:
        eid = entry["id"]
        defaults = schema_defaults(schema, eid)
        modes = [("random", "high")] + (
            [("extreme", "low"), ("extreme", "high")] if entry["extreme"] else []
        )

        def req(values=None, mode="random", edge="high", reroll=0, eid=eid) -> PaneRequest:
            return PaneRequest("", eid, values or {}, mode, reroll, edge, profile)

        for mode, edge in modes:
            add(req(None, mode, edge), defaults)  # the profile's own settings
        for knob in entry["knobs"]:
            if knob.get("share") and knob["kind"] in ("scalar", "range"):
                for stop in knob_stops(knob, stops):
                    for mode, edge in modes:
                        add(req({knob["key"]: stop}, mode, edge), defaults)
            elif knob["kind"] == "enum":
                for value in _enum_values(knob):
                    add(req({knob["key"]: value}), defaults)
        for k in range(1, rerolls + 1):
            add(req(reroll=k), defaults)
    for name in PROFILES:
        for k in range(0, rerolls + 1):
            add(PaneRequest("", f"combined.{name}", {}, "random", k, "high", profile), {})
    return list(seen.values())


def choose_grid(
    schema: Dict[str, Any],
    n_clips: int,
    bytes_per_pane: float,
    budget_bytes: float,
    fixed_bytes: float = 0,
) -> Tuple[int, int]:
    """
    `(stops, rerolls)`: the first of `GRIDS` whose estimated size fits `budget_bytes`.

    Size = `fixed_bytes` (frames, page, base blobs) + clips x panes x `bytes_per_pane`. When none
    fits, `(3, 0)` with a warning.
    """
    for stops, rerolls in GRIDS:
        panes = len(plan_grid(schema, stops, rerolls))
        if fixed_bytes + n_clips * panes * bytes_per_pane <= budget_bytes:
            return stops, rerolls
    log.warning("no grid fits the budget; using (3, 0), the smallest")
    return GRIDS[-1]


# ---- the mesh --------------------------------------------------------------------------------
def decimate_mesh(
    vertices: np.ndarray, faces: np.ndarray, target: int = 3000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadric decimation of a rest mesh to about `target` vertices (twice that many faces).

    Returns `(index, faces)`: `index` maps each kept vertex to its nearest original vertex (distinct,
    no two kept vertices share one), so a
    frame's decimated mesh is `verts[index]` (or `verts[:, index]` for a clip), and `faces` index
    the kept vertices.
    """
    import trimesh
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=np.float64)
    mesh = trimesh.Trimesh(vertices, np.asarray(faces), process=False)
    small = mesh.simplify_quadric_decimation(face_count=2 * target)
    tree = cKDTree(vertices)
    kept = np.asarray(small.vertices)
    _, near = tree.query(kept, k=min(16, len(vertices)))
    index = np.empty(len(kept), dtype=np.int64)
    used = set()
    for i, candidates in enumerate(near):  # nearest original vertex not already taken
        pick = next((int(c) for c in candidates if int(c) not in used), None)
        if pick is None:  # all 16 nearest taken (never seen): widen the search
            order = np.argsort(np.linalg.norm(vertices - kept[i], axis=1))
            pick = next(int(c) for c in order if int(c) not in used)
        used.add(pick)
        index[i] = pick
    return index, np.asarray(small.faces, dtype=np.uint16)


def _share_mesh(meta, arrays, index, faces=None, mesh_mm=MESH_MM):
    """The pane with `mesh_verts` reduced to `index` and quantised to `mesh_mm` millimetres."""
    arrays = dict(arrays)
    verts, dtype = arrays["mesh_verts"]
    verts = np.asarray(verts)[:, index]
    unit = 1000.0 / mesh_mm
    ratio = SCALE_I16 / unit
    arrays["mesh_verts"] = (np.round(verts / ratio).astype(np.int16), dtype)
    if faces is not None:
        arrays["mesh_faces"] = (faces, "uint16")
    return dict(meta, scale_i16=unit), arrays


# ---- the build -------------------------------------------------------------------------------
def _gzip(blob: bytes) -> bytes:
    return gzip.compress(blob, compresslevel=9, mtime=0)


def _script(sha1: str, gz: bytes) -> str:
    return f'Lab.addBlob("{sha1}","{base64.b64encode(gz).decode("ascii")}");\n'


def _profile_path(name: str) -> Path:
    return CONFIGS_DIR / f"slr_week4_{name}.yaml"


def _rest_mesh(prep) -> np.ndarray:
    ok = np.asarray(prep.base_meta["body_ok"], bool)
    frame = int(np.argmax(ok)) if ok.any() else 0
    return prep.base_arrays["mesh_verts"][0][frame].astype(np.float64) / SCALE_I16


def _frame_count(store, clip_id: str) -> int:
    n = 0
    while True:
        try:
            store.frame_path(clip_id, n)
        except KeyError:
            return n
        n += 1


def build_share(
    clip_ids: List[str],
    store,
    models,
    out_zip: Path,
    budget_mb: int = 200,
    profile: str = "default",
    grid: Optional[Tuple[int, int]] = None,
    mesh_target: int = MESH_VERTICES,
    mesh_mm: float = MESH_MM,
    entries: Optional[List[str]] = None,
    progress: Callable[[str], None] = lambda text: None,
) -> Dict[str, Any]:
    """
    Run the grid on every clip and write the share zip `out_zip`. Returns a summary
    `{panes, unique_blobs, size_bytes, stops, rerolls, failed, ...}`.

    `grid` forces `(stops, rerolls)` (tests); otherwise `choose_grid` picks the richest grid whose
    estimate fits `budget_mb`, from one measured pane per clip. `entries` limits the panes to
    those entry ids (tests; `combined.<p>` names a whole profile).
    """
    out_zip = Path(out_zip)
    base_profile: Profile = load_profile(_profile_path(profile))
    schema = build_schema(base_profile)
    bases = {name: load_profile(_profile_path(name)) for name in PROFILES}
    progress("preparing clips")
    preps = {cid: store.get(cid) for cid in clip_ids}
    first = preps[clip_ids[0]]
    index, faces = decimate_mesh(_rest_mesh(first), first.base_arrays["mesh_faces"][0], mesh_target)

    def run(prep, req: PaneRequest):
        base = bases[req.entry_id[len("combined.") :]] if req.whole else base_profile
        meta, arrays = run_pane(prep, req, base, models)
        return _share_mesh(meta, arrays, index, None, mesh_mm)

    base_blobs: Dict[str, bytes] = {}
    for cid, prep in preps.items():
        meta, arrays = _share_mesh(prep.base_meta, prep.base_arrays, index, faces, mesh_mm)
        base_blobs[cid] = encode_pane(meta, arrays)
    frames = {cid: _frame_count(store, cid) for cid in clip_ids}
    frame_bytes = sum(
        store.frame_path(c, i).stat().st_size for c in clip_ids for i in range(frames[c])
    )
    page_bytes = sum(p.stat().st_size for p in STATIC_DIR.rglob("*") if p.is_file())
    base_gz = {cid: _gzip(blob) for cid, blob in base_blobs.items()}  # gzipped once
    fixed = frame_bytes + page_bytes + sum(len(g) for g in base_gz.values())

    if grid is None:
        sizes = []
        probe = PaneRequest("", "body.joint_coupled_noise", {}, "random", 0, "high", profile)
        for cid, prep in preps.items():
            sizes.append(
                len(_gzip(encode_pane(*run(prep, dataclasses.replace(probe, clip_id=cid)))))
            )
        per_pane = GZIP_FACTOR * float(np.mean(sizes))
        grid = choose_grid(schema, len(clip_ids), per_pane, budget_mb * 1e6, fixed)
        progress(f"grid {grid}, about {per_pane / 1e3:.0f} kB a pane")
    stops, rerolls = grid
    requests = plan_grid(schema, stops, rerolls, profile)
    if entries is not None:
        requests = [r for r in requests if r.entry_id in entries]

    stage = Path(tempfile.mkdtemp(prefix="lab_share_"))
    try:
        data = stage / "data"
        panes: Dict[str, str] = {}
        failed: List[str] = []
        unique = set()
        for cid, prep in preps.items():
            (data / cid / "frames").mkdir(parents=True)
            for i in range(frames[cid]):
                shutil.copy2(store.frame_path(cid, i), data / cid / "frames" / f"{i:04d}.jpg")

        def write_blob(cid: str, blob: bytes, gz: Optional[bytes] = None) -> str:
            """Write `data/<cid>/<sha1>.js` once per (clip, blob); `gz` is the blob if gzipped."""
            sha1 = hashlib.sha1(blob).hexdigest()
            if (cid, sha1) not in unique:
                unique.add((cid, sha1))
                gz = _gzip(blob) if gz is None else gz
                (data / cid / f"{sha1}.js").write_text(_script(sha1, gz), encoding="ascii")
            return sha1

        base_ids = {cid: write_blob(cid, blob, base_gz[cid]) for cid, blob in base_blobs.items()}
        for cid, prep in preps.items():
            for n, tmpl in enumerate(requests):
                req = dataclasses.replace(tmpl, clip_id=cid)
                try:
                    blob = encode_pane(*run(prep, req))
                except ValueError as exc:  # a combination the augmentation rejects: not shared
                    failed.append(f"{cid} {req.key()}: {exc}")
                    log.warning("skipped %s", failed[-1])
                    continue
                defaults = schema_defaults(schema, req.entry_id)
                panes[req.share_key(defaults)] = write_blob(cid, blob)
                if n % 50 == 0:
                    progress(f"clip {cid}: pane {n + 1}/{len(requests)}")

        lab_data = {
            "profile": profile,
            "schema": jsonable(schema),
            "profileText": _profile_path(profile).read_text(),
            "clips": [
                {"id": c, "n": preps[c].n_images, "fps": preps[c].clip.fps} for c in clip_ids
            ],
            "base": base_ids,
            "panes": panes,
            "stops": stops_table(schema, stops),
            "rerolls": rerolls,
        }
        (data / "manifest.js").write_text(
            "window.LAB_DATA = " + json.dumps(jsonable(lab_data), separators=(",", ":")) + ";\n",
            encoding="utf-8",
        )
        # Only the manifest loads at page open; the page adds each blob's script on first use.
        (data / "load.js").write_text(
            "// Written by the share build: the manifest. Blob scripts load when first needed.\n"
            "document.write('<script src=\"data/manifest.js\"><\\/script>');\n",
            encoding="utf-8",
        )
        size = _write_zip(stage, out_zip)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    summary = {
        "panes": len(panes),
        "unique_blobs": len(unique),
        "size_bytes": size,
        "stops": stops,
        "rerolls": rerolls,
        "clips": list(clip_ids),
        "mesh_vertices": int(len(index)),
        "failed": failed,
    }
    if size > budget_mb * 1e6:
        log.warning("zip is %.1f MB, over the %d MB budget", size / 1e6, budget_mb)
    return summary


def _write_zip(stage: Path, out_zip: Path) -> int:
    """Zip `static/` (with the staged `data/` over it) under `augmentation-lab/`; return its size."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    staged = {p.relative_to(stage).as_posix(): p for p in stage.rglob("*") if p.is_file()}
    files: Dict[str, Path] = {}
    for p in sorted(STATIC_DIR.rglob("*")):
        rel = p.relative_to(STATIC_DIR).as_posix()
        if p.is_file() and p.name != "index.md" and "__pycache__" not in rel:
            files[rel] = p
    files.update(staged)  # data/load.js replaces the placeholder
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(f"{ROOT}/README.txt", README)
        for rel, path in files.items():
            if rel.lower().endswith(FORBIDDEN_SUFFIXES):
                raise ValueError(
                    f"refusing to pack {rel}: model and cache files stay out of the zip"
                )
            if rel.endswith(".jpg"):  # already compressed
                zf.write(path, f"{ROOT}/{rel}", compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(path, f"{ROOT}/{rel}")
    return out_zip.stat().st_size

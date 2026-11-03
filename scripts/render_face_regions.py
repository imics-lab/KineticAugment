#!/usr/bin/env python3
"""
Render the head-region segmentation (V1.3 design §5) for visual review: front and side views of
the SMPL-X head vertices, coloured by region, brows and nose outlined by subregion.

Usage:
    python scripts/render_face_regions.py [--output docs/media/face_regions.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kinetic_augment.body_model.assembly import pose_mesh  # noqa: E402
from kinetic_augment.body_model.face_correspondence import load_face_correspondence  # noqa: E402
from kinetic_augment.body_model.face_regions import (  # noqa: E402
    REGIONS, SUBREGIONS, build_head_surface, vertex_normals,
)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/media/face_regions.png"))
    args = parser.parse_args()
    skeleton = SMPLXSkeleton("models")
    surface = build_head_surface(skeleton, load_face_correspondence("models"))
    rest = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3)).vertices
    points = rest[surface.vertex_ids]
    normals = vertex_normals(rest, np.asarray(skeleton.model.faces, np.int64))[surface.vertex_ids]
    colours = plt.get_cmap("tab10")(surface.region)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, (axis, facing, title) in zip(axes, [(0, 2, "front"), (2, 0, "side (left)")]):
        visible = normals[:, facing] > 0
        order = np.argsort(points[visible, facing])
        p, c = points[visible][order], colours[visible][order]
        ax.scatter(p[:, axis], p[:, 1], c=c, s=4)
        sub = surface.subregion[visible][order]
        ax.scatter(p[sub == SUBREGIONS.index("Brow"), axis], p[sub == SUBREGIONS.index("Brow"), 1],
                   s=1, c="k")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=plt.get_cmap("tab10")(i), label=name)
               for i, name in enumerate(REGIONS)]
    fig.legend(handles=handles, loc="lower center", ncol=5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

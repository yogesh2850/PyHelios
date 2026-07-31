"""Open a trained Gaussian-splat PLY as an interactive 3D point cloud.

Opens a native window (mouse: left-drag orbit, scroll zoom, right-drag pan) --
needs a display (DISPLAY / Wayland) on the machine running this script.

Run from the gsplat environment:
    python view_apple_tree_pointcloud.py
"""

import argparse
import os
import sys

import numpy as np
import pyvista as pv

from render_apple_tree_splats import DEFAULT_PLY, read_gaussian_ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", default=DEFAULT_PLY, help="Gaussian-splat PLY path")
    parser.add_argument("--point-size", type=float, default=4.0)
    args = parser.parse_args()

    if not os.path.isfile(args.ply):
        raise FileNotFoundError(
            f"Gaussian PLY not found: {args.ply}\n"
            "Run apple_tree_gaussian_splatting.py first or pass --ply PATH."
        )

    means, colors, scales, quats, opacities = read_gaussian_ply(args.ply)
    print(f"Loaded {len(means)} Gaussian centers from {args.ply}")

    cloud = pv.PolyData(means)
    cloud["colors"] = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

    plotter = pv.Plotter(title=f"Apple orchard Gaussian centers ({len(means)} points)")
    plotter.add_points(
        cloud, scalars="colors", rgb=True, point_size=args.point_size,
        render_points_as_spheres=True,
    )
    plotter.set_background("lightskyblue")
    plotter.show_axes()
    plotter.show()


if __name__ == "__main__":
    main()

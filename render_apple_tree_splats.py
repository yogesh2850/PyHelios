"""Render a trained Gaussian-splat PLY without rebuilding the PyHelios scene.

Run from the gsplat environment:
    python render_apple_tree_splats.py

The script writes a turntable preview to:
    renders/gaussian_splatting/viewer/

Use --show to ask the operating system to open the first rendered PNG.
"""

import argparse
import math
import os
import struct
import subprocess
import sys

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

SH0_C0 = 0.28209479177387814
DEFAULT_PLY = os.path.join(
    "renders", "gaussian_splatting", "apple_orchard_splats.ply"
)
DEFAULT_OUTPUT = os.path.join("renders", "gaussian_splatting", "viewer")


def read_gaussian_ply(path):
    """Read the binary little-endian PLY written by gsplat.export_splats."""
    with open(path, "rb") as ply_file:
        header_lines = []
        while True:
            line = ply_file.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("This viewer expects a binary little-endian PLY")

        vertex_count = None
        properties = []
        in_vertex_element = False
        for line in header_lines:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex_element = True
            elif parts[:2] == ["element", "face"]:
                in_vertex_element = False
            elif in_vertex_element and parts[:1] == ["property"]:
                if len(parts) != 3 or parts[1] != "float":
                    raise ValueError(f"Unsupported PLY property: {line}")
                properties.append(parts[2])

        if vertex_count is None:
            raise ValueError("PLY does not contain a vertex element")

        dtype = np.dtype([(name, "<f4") for name in properties])
        vertices = np.fromfile(ply_file, dtype=dtype, count=vertex_count)

    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = required.difference(vertices.dtype.names or ())
    if missing:
        raise ValueError(f"PLY is missing Gaussian fields: {sorted(missing)}")

    means = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    sh0 = np.column_stack([
        vertices[name] for name in ("f_dc_0", "f_dc_1", "f_dc_2")
    ])
    colors = np.clip(sh0 * SH0_C0 + 0.5, 0.0, 1.0)
    scales = np.column_stack([
        vertices[name] for name in ("scale_0", "scale_1", "scale_2")
    ])
    quats = np.column_stack([
        vertices[name] for name in ("rot_0", "rot_1", "rot_2", "rot_3")
    ])
    opacities = vertices["opacity"]

    return means, colors, scales, quats, opacities


def look_at_view_matrix(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Create an OpenCV-convention world-to-camera matrix for gsplat."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray(world_up, dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    camera_to_world = np.stack([right, down, forward], axis=1)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = camera_to_world.T
    viewmat[:3, 3] = -camera_to_world.T @ eye
    return viewmat


def render_splat(means, colors, scales, quats, opacities, eye, target,
                 width, height, fov_deg, background, device):
    """Render one Gaussian-splat view with gsplat."""
    means_t = torch.from_numpy(means).to(device=device, dtype=torch.float32)
    colors_t = torch.from_numpy(colors).to(device=device, dtype=torch.float32)
    scales_t = torch.exp(torch.from_numpy(scales).to(device=device, dtype=torch.float32))
    quats_t = torch.from_numpy(quats).to(device=device, dtype=torch.float32)
    quats_t = torch.nn.functional.normalize(quats_t, dim=-1)
    opacities_t = torch.sigmoid(
        torch.from_numpy(opacities).to(device=device, dtype=torch.float32)
    )

    focal = height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = torch.tensor(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    viewmat = torch.from_numpy(
        look_at_view_matrix(eye, target)
    ).to(device=device).unsqueeze(0)
    background_t = torch.tensor(
        background, device=device, dtype=torch.float32
    ).reshape(1, 3)

    with torch.no_grad():
        render, _, _ = rasterization(
            means=means_t,
            quats=quats_t,
            scales=scales_t,
            opacities=opacities_t,
            colors=colors_t,
            viewmats=viewmat,
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=None,
            backgrounds=background_t,
            packed=False,
        )
    image = (render[0].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", default=DEFAULT_PLY, help="Gaussian-splat PLY path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="PNG output directory")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--views", type=int, default=12, help="Number of orbit images")
    parser.add_argument("--fov", type=float, default=45.0, help="Vertical field of view")
    parser.add_argument("--show", action="store_true", help="Open the first PNG after rendering")
    args = parser.parse_args()

    if not os.path.isfile(args.ply):
        raise FileNotFoundError(
            f"Gaussian PLY not found: {args.ply}\n"
            "Run apple_tree_gaussian_splatting.py first or pass --ply PATH."
        )
    if args.views < 1 or args.width < 1 or args.height < 1:
        raise ValueError("views, width, and height must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.ply}")
    means, colors, scales, quats, opacities = read_gaussian_ply(args.ply)
    print(f"Loaded {len(means)} Gaussians on {device}")

    center = means.mean(axis=0).astype(np.float32)
    radius = float(np.linalg.norm(means - center, axis=1).max())
    radius = max(radius, 1e-3)
    distance = radius * 1.35 / math.sin(math.radians(args.fov / 2.0))
    background = (0.70, 0.85, 1.0)
    os.makedirs(args.output, exist_ok=True)

    for index in range(args.views):
        azimuth = 2.0 * math.pi * index / args.views
        elevation = math.radians(18.0)
        eye = center + distance * np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ], dtype=np.float32)
        image = render_splat(
            means, colors, scales, quats, opacities,
            eye, center, args.width, args.height, args.fov,
            background, device,
        )
        path = os.path.join(args.output, f"view_{index:03d}.png")
        image.save(path)
        print(f"Saved {path}")

    first_image = os.path.abspath(os.path.join(args.output, "view_000.png"))
    print(f"Done. Open the rendered images in {os.path.abspath(args.output)}")
    if args.show:
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", first_image])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", first_image])
        elif os.name == "nt":
            os.startfile(first_image)


if __name__ == "__main__":
    main()

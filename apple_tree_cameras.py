import math
import os
import sys

from apple_tree import build_apple_tree
from pyhelios import Context, PlantArchitecture, Visualizer
from pyhelios.types import RGBcolor, vec3

CAMERA_FOV_DEG = 45.0    # matches Visualizer's default vertical FOV (not overridden below)
CAMERA_MARGIN = 1.3      # headroom so the tree doesn't touch the frame edges
CAMERA_DISTANCE_MIN = 1.0  # floor for very young/small trees

CAMERA_RIGS = [
    {"name": "above", "height_frac": 1.15},
    {"name": "level", "height_frac": 0.5},
    {"name": "below", "height_frac": 0.05},
]

OUTPUT_DIR = "renders"


def camera_position_for_tree(context, plantarch, plant_id, base_position, rig, aspect_ratio):
    """Compute a camera position/lookAt in front of one tree for a given rig.

    Distance is sized from the tree's own bounding box (height and width),
    not a fixed offset -- a fixed distance either crops a tree that grew
    larger than expected or pulls neighboring trees into frame once they're
    wide enough to cross the horizontal FOV at that distance.
    """
    uuids = plantarch.getAllPlantUUIDs(plant_id)
    x_bounds, y_bounds, z_bounds = context.getDomainBoundingBox(uuids)
    tree_height = z_bounds.y - z_bounds.x
    tree_width = x_bounds.y - x_bounds.x

    half_fov_v = math.radians(CAMERA_FOV_DEG) / 2
    half_fov_h = math.atan(aspect_ratio * math.tan(half_fov_v))
    distance_for_height = (tree_height / 2) / math.tan(half_fov_v) * CAMERA_MARGIN
    distance_for_width = (tree_width / 2) / math.tan(half_fov_h) * CAMERA_MARGIN
    distance = max(distance_for_height, distance_for_width, CAMERA_DISTANCE_MIN)

    look_at = vec3(base_position.x, base_position.y, z_bounds.x + 0.5 * tree_height)
    position = vec3(
        base_position.x,
        base_position.y - distance,
        z_bounds.x + rig["height_frac"] * tree_height,
    )
    return position, look_at


if __name__ == "__main__":
    AGE_DAYS = 720.0
    POSITIONS = [vec3(0, 0, 0), vec3(1.5, 0, 0), vec3(3, 0, 0)]
    WIDTH, HEIGHT = 1000, 800

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with Context() as context:
        with PlantArchitecture(context) as plantarch:
            plant_ids = []
            for i, position in enumerate(POSITIONS):
                plant_id = build_apple_tree(plantarch, position=position, age_days=AGE_DAYS)
                plant_ids.append(plant_id)
                print(f"Built apple tree plant_id={plant_id} at {position}")

            all_uuids = []
            for plant_id in plant_ids:
                all_uuids.extend(plantarch.getAllPlantUUIDs(plant_id))

            with Visualizer(width=WIDTH, height=HEIGHT, headless=True) as visualizer:
                visualizer.buildContextGeometry(context, uuids=all_uuids)
                visualizer.setBackgroundColor(RGBcolor(0.70, 0.85, 1.0))
                visualizer.setLightingModel("phong_shadowed")

                for plant_id, position in zip(plant_ids, POSITIONS):
                    for rig in CAMERA_RIGS:
                        camera_pos, look_at = camera_position_for_tree(
                            context, plantarch, plant_id, position, rig, WIDTH / HEIGHT
                        )
                        visualizer.setCameraPosition(position=camera_pos, lookAt=look_at)
                        visualizer.plotUpdate()

                        filename = os.path.join(
                            OUTPUT_DIR, f"tree{plant_id}_{rig['name']}.png"
                        )
                        visualizer.printWindow(filename)
                        print(f"  Saved {filename}")

    print(f"\nDone. Images written to {OUTPUT_DIR}/")

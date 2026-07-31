from pyhelios import Context, Visualizer
from pyhelios.Visualizer import VisualizerError
from pyhelios.types import *

context = Context()
uuid = context.addPatch(
        center=vec3(2, 0, 0),
        size=vec2(1, 1),
        color=RGBcolor(1, 0, 0)
    )

uuis2 = context.addTriangle(
    vertex0=vec3(3, 0, 0),
    vertex1=vec3(1, 1, 1),
    vertex2=vec3(0, 1, 0),
    color=RGBcolor(0, 1, 0))

print(f"Patch UUID: {uuid}")
print(f"Triangle UUID: {uuis2}")

uuis3 = context.addSphere(
    center=vec3(0, 0, 0),
    radius=1,
    color=RGBcolor(0, 0, 1)
)

print(f"Sphere UUID: {uuis3}")

with Visualizer(800, 600) as visualizer:
    visualizer.buildContextGeometry(context)
    visualizer.setBackgroundColor(RGBcolor(0.1, 0.1, 0.15))
    visualizer.setLightDirection(vec3(1, 1, -1))
    visualizer.setLightingModel("phong")
    visualizer.setCameraPosition(vec3(4, 4, 3), vec3(0, 0, 0))
    visualizer.plotInteractive()
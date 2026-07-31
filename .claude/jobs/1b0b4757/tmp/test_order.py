import sys, contextlib, io
sys.path.insert(0, "/home/yogesh/PyHelios/.claude/worktrees/phase1-groundtruth")
from pyhelios import Context, PlantArchitecture
from pyhelios.types import vec3

def try_order(call_before_build):
    with Context() as context:
        with PlantArchitecture(context) as plantarch:
            if call_before_build:
                plantarch.optionalOutputObjectData(["plantID", "fruitID", "leafID", "rank", "age", "phenology_stage"])
            plantarch.loadPlantModelFromLibrary("apple")
            plant_id = plantarch.buildPlantInstanceFromLibrary(base_position=vec3(0,0,0), age=720.0)
            if not call_before_build:
                plantarch.optionalOutputObjectData(["plantID", "fruitID", "leafID", "rank", "age", "phenology_stage"])
            uuids = plantarch.getAllPlantUUIDs(plant_id)
            fruit_uuids = context.filterPrimitivesByData(uuids, "object_label", "fruit")
            if not fruit_uuids:
                print(f"order=before:{call_before_build} -- NO FRUIT PRIMITIVES FOUND ({len(uuids)} total prims)")
                return
            fruit_objs = context.getUniquePrimitiveParentObjectIDs(fruit_uuids, include_zero=False)
            print(f"order=before:{call_before_build} -- {len(fruit_uuids)} fruit prims, {len(fruit_objs)} fruit objects")
            oid = fruit_objs[0]
            labels = context.listObjectData(oid)
            print(f"  listObjectData(obj {oid}) = {labels}")
            for want in ["plantID", "fruitID", "rank", "age", "phenology_stage"]:
                present = want in labels
                print(f"  has {want}: {present}")

print("=== Test: optionalOutputObjectData BEFORE build ===")
try_order(True)
print()
print("=== Test: optionalOutputObjectData AFTER build ===")
try_order(False)

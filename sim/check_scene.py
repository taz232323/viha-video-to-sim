from pathlib import Path

import mujoco


SCENE_PATH = Path(__file__).resolve().parent / "worlds" / "table_bowl_pick_place.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    for _ in range(240):
        mujoco.mj_step(model, data)

    print(f"Loaded: {SCENE_PATH}")
    print(f"Bodies: {model.nbody}")
    print(f"Geoms: {model.ngeom}")
    print(f"Joints: {model.njnt}")
    print(f"Actuators: {model.nu}")
    print(f"Sim time after check: {data.time:.3f}s")


if __name__ == "__main__":
    main()


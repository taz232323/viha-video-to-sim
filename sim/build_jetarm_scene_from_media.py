from __future__ import annotations

import argparse
from pathlib import Path

from build_fanuc_scene_from_media import main as shared_main


JETARM_BASE_SCENE = Path(__file__).resolve().parent / "worlds" / "table_bowl_pick_place_jetarm.xml"


if __name__ == "__main__":
    # The shared media builder is robot-agnostic when the spec supplies
    # robot.base_body and calibration.robot_base_world_xyz. This wrapper gives
    # JetArm callers a clearer command name and default base scene.
    import sys

    if "--base-scene" not in sys.argv:
        sys.argv.extend(["--base-scene", str(JETARM_BASE_SCENE)])
    if "--runner-script" not in sys.argv:
        sys.argv.extend(["--runner-script", str(Path(__file__).resolve().parent / "run_jetarm_pick_place_demo.py")])
    shared_main()

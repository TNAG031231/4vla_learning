#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from nuscenes.nuscenes import NuScenes
except ModuleNotFoundError as error:
    if error.name == "nuscenes":
        raise SystemExit(
            "nuscenes-devkit is not installed. Install it with:\n"
            "pip install nuscenes-devkit"
        ) from error
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a nuScenes sample_token to CAM_FRONT image mapping."
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("data/nuscenes"),
        help="nuScenes dataset root (default: data/nuscenes).",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes dataset version (default: v1.0-mini).",
    )
    parser.add_argument(
        "--sample-token",
        help="Sample token to inspect (default: first sample of the first scene).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    dataroot = arguments.dataroot
    nuscenes = NuScenes(
        version=arguments.version,
        dataroot=str(dataroot),
        verbose=False,
    )

    first_scene = nuscenes.scene[0]
    sample_token = arguments.sample_token or first_scene["first_sample_token"]
    sample = nuscenes.get("sample", sample_token)
    scene = nuscenes.get("scene", sample["scene_token"])
    cam_front_token = sample["data"]["CAM_FRONT"]
    cam_front = nuscenes.get("sample_data", cam_front_token)
    relative_image_path = Path(cam_front["filename"])
    image_exists = (dataroot / relative_image_path).is_file()

    print(f"Scene count: {len(nuscenes.scene)}")
    print(f"sample_token: {sample['token']}")
    print(f"scene_token: {scene['token']}")
    print(f"timestamp: {sample['timestamp']}")
    print(f"CAM_FRONT sample_data_token: {cam_front['token']}")
    print(f"CAM_FRONT filename: {cam_front['filename']}")
    print(f"CAM_FRONT relative image path: {relative_image_path.as_posix()}")
    print(f"CAM_FRONT absolute image exists: {str(image_exists).lower()}")


if __name__ == "__main__":
    main()

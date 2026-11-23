import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

from SurroundOcc.tools.infer import (
    CAMERA_ORDER,
    SurroundOccRunner,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
REAL_DATA_DIR = REPOSITORY_ROOT / "real_data"
DEFAULT_FRAME_FILE = REAL_DATA_DIR / "frames.json"
if not DEFAULT_FRAME_FILE.is_file():
    DEFAULT_FRAME_FILE = REAL_DATA_DIR / "frame.json"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "SurroundOcc" / "projects" / "configs" /
    "surroundocc" / "surroundocc_inference.py"
)


def load_frame_data(frame_file):
    """Load image paths and fixed camera calibration from a frame JSON."""
    frame_file = Path(frame_file).expanduser().resolve()
    if not frame_file.is_file():
        raise FileNotFoundError(
            "Frame JSON does not exist: {}".format(frame_file))

    with frame_file.open("r") as stream:
        frame = json.load(stream)

    cameras = frame.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("Frame JSON must contain a 'cameras' object")
    missing = [camera for camera in CAMERA_ORDER if camera not in cameras]
    extra = [camera for camera in cameras if camera not in CAMERA_ORDER]
    if missing or extra:
        raise ValueError(
            "Frame JSON must describe exactly the six model cameras. "
            "Missing: {}; unexpected: {}".format(missing, extra))

    calibrations = {}
    images = {}
    for camera in CAMERA_ORDER:
        camera_data = cameras[camera]
        if "intrinsic" not in camera_data:
            raise ValueError("{} has no intrinsic matrix".format(camera))
        if "camera_to_lidar" not in camera_data:
            raise ValueError("{} has no camera_to_lidar transform".format(
                camera))
        if "image" not in camera_data:
            raise ValueError("{} has no image path".format(camera))

        image_path = Path(camera_data["image"]).expanduser()
        if not image_path.is_absolute():
            image_path = frame_file.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            images_dir = frame_file.parent / "images"
            fallback_images = [
                images_dir / (camera + extension)
                for extension in (".jpg", ".jpeg", ".png", ".bmp")
                if (images_dir / (camera + extension)).is_file()
            ]
            if len(fallback_images) == 1:
                warnings.warn(
                    "{} path in frame JSON does not exist; using {}".format(
                        camera, fallback_images[0]))
                image_path = fallback_images[0]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                "{} image could not be read: {}".format(camera, image_path))

        calibrations[camera] = {
            "intrinsic": camera_data["intrinsic"],
            "camera_to_lidar": camera_data["camera_to_lidar"],
        }
        images[camera] = np.ascontiguousarray(image)

    frame_id = str(frame.get("frame_id", frame_file.stem))
    return frame_id, calibrations, images


def run_inference(checkpoint, frame_file=DEFAULT_FRAME_FILE,
                  config=DEFAULT_CONFIG):
    """Load a frame and run one SurroundOcc inference pass."""
    frame_id, calibrations, images = load_frame_data(frame_file)
    runner = SurroundOccRunner(
        checkpoint_path=checkpoint,
        calibrations=calibrations,
        config_path=config,
    )
    return runner.infer(images, frame_id=frame_id)


def main():
    parser = argparse.ArgumentParser(
        description="Run SurroundOcc on the cameras described in a frame JSON.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained SurroundOcc occupancy checkpoint.")
    parser.add_argument(
        "--frame-json",
        default=str(DEFAULT_FRAME_FILE),
        help="Frame JSON; defaults to the existing file in real_data/.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="SurroundOcc inference config.")
    parser.add_argument(
        "--output",
        default="result/occupancy_prediction.npz",
        help="Path for the returned occupancy arrays.")
    args = parser.parse_args()

    # Load the frame and calibrations
    frame_id, calibrations, images = load_frame_data(args.frame_json)
    runner = SurroundOccRunner(
        checkpoint_path=args.checkpoint,
        calibrations=calibrations,
        config_path=args.config,
    )

    # Run inference
    prediction = runner.infer(images, frame_id=frame_id)

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        occupancy=prediction.occupancy,
        confidence=prediction.confidence,
        semantic=np.asarray(prediction.semantic),
        point_cloud_range=np.asarray(prediction.point_cloud_range),
        class_names=np.asarray(prediction.class_names),
    )
    print("Frame: {}".format(frame_id))
    print("Saved occupancy prediction: {}".format(output_path.resolve()))
    print("Voxel grid shape: {}".format(prediction.occupancy.shape))


if __name__ == "__main__":
    main()

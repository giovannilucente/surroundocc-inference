import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image
from matplotlib.colors import BoundaryNorm, ListedColormap


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_FRAME_JSON = REPOSITORY_ROOT / "real_data" / "frame.json"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


CLASS_COLORS = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],
        [255, 192, 203, 255],
        [255, 255, 0, 255],
        [0, 150, 245, 255],
        [0, 255, 255, 255],
        [200, 180, 0, 255],
        [255, 0, 0, 255],
        [255, 240, 150, 255],
        [135, 60, 0, 255],
        [160, 32, 240, 255],
        [255, 0, 255, 255],
        [139, 137, 137, 255],
        [75, 0, 75, 255],
        [150, 240, 80, 255],
        [230, 230, 250, 255],
        [0, 175, 0, 255],
        [0, 255, 127, 255],
        [255, 99, 71, 255],
        [0, 191, 255, 255],
    ],
    dtype=np.uint8,
)
CLASS_NAMES = (
    "empty",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "ego car",
    "class 18",
    "class 19",
)


def load_prediction(path):
    """Load and validate the compressed prediction written by the runner."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Prediction file does not exist: {}".format(path))

    with np.load(str(path), allow_pickle=False) as data:
        required = {"occupancy", "semantic", "point_cloud_range"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                "Prediction archive is missing fields: {}".format(
                    ", ".join(sorted(missing))))
        occupancy = np.array(data["occupancy"])
        semantic = bool(np.asarray(data["semantic"]).item())
        point_cloud_range = np.asarray(
            data["point_cloud_range"], dtype=np.float32)
        class_names = tuple(
            str(name) for name in data["class_names"]
        ) if "class_names" in data.files else CLASS_NAMES

    if occupancy.ndim != 3:
        raise ValueError(
            "Expected a 3D occupancy grid, got shape {}".format(
                occupancy.shape))
    if point_cloud_range.shape != (6,):
        raise ValueError(
            "point_cloud_range must contain six values, got shape {}".format(
                point_cloud_range.shape))
    if not np.all(np.isfinite(point_cloud_range)):
        raise ValueError("point_cloud_range contains non-finite values")
    if not class_names:
        raise ValueError("Prediction archive contains no class names")
    if len(class_names) > len(CLASS_COLORS):
        raise ValueError(
            "Prediction defines {} classes, but the palette supports {}"
            .format(len(class_names), len(CLASS_COLORS)))

    return occupancy, semantic, point_cloud_range, class_names


def occupancy_to_points(occupancy, semantic, point_cloud_range, threshold):
    """Convert occupied voxels to their world-space centers and class labels."""
    if semantic:
        occupied = occupancy > 0
        labels = occupancy[occupied].astype(np.int32, copy=False)
    else:
        if not np.issubdtype(occupancy.dtype, np.number):
            raise ValueError("Binary occupancy values must be numeric")
        if not np.all(np.isfinite(occupancy)):
            raise ValueError("Binary occupancy contains non-finite values")
        occupied = occupancy >= threshold
        labels = np.ones(int(occupied.sum()), dtype=np.int32)

    indices = np.argwhere(occupied).astype(np.float32)
    if indices.size == 0:
        raise ValueError(
            "The prediction contains no occupied voxels at this threshold")
    if np.any(labels < 0) or np.any(labels >= len(CLASS_COLORS)):
        raise ValueError(
            "Prediction contains class IDs outside the visualization palette")

    lower = point_cloud_range[:3]
    upper = point_cloud_range[3:]
    voxel_size = (upper - lower) / np.asarray(occupancy.shape, dtype=np.float32)
    points = lower + (indices + 0.5) * voxel_size
    return points, labels, voxel_size


def load_camera_metadata(frame_json):
    """Load camera calibration and image dimensions from the frame JSON."""
    frame_json = Path(frame_json).expanduser().resolve()
    if not frame_json.is_file():
        raise FileNotFoundError(
            "Frame JSON does not exist: {}".format(frame_json))

    with frame_json.open("r") as stream:
        frame = json.load(stream)
    cameras = frame.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("Frame JSON must contain a 'cameras' object")
    missing = sorted(set(CAMERA_ORDER) - set(cameras))
    if missing:
        raise ValueError(
            "Frame JSON is missing camera calibration for: {}".format(
                ", ".join(missing)))

    camera_data = {}
    for camera_name in CAMERA_ORDER:
        camera = cameras[camera_name]
        intrinsic = np.asarray(camera.get("intrinsic"), dtype=np.float64)
        camera_to_lidar = np.asarray(
            camera.get("camera_to_lidar"), dtype=np.float64)
        if intrinsic.shape != (3, 3) or camera_to_lidar.shape != (4, 4):
            raise ValueError(
                "{} must provide a 3x3 intrinsic and 4x4 "
                "camera_to_lidar transform".format(camera_name))
        if not np.all(np.isfinite(intrinsic)) or not np.all(
                np.isfinite(camera_to_lidar)):
            raise ValueError(
                "{} calibration contains non-finite values".format(
                    camera_name))

        image_path = Path(camera.get("image", "")).expanduser()
        if not image_path.is_absolute():
            image_path = frame_json.parent / image_path
        if not image_path.is_file():
            fallback_images = [
                frame_json.parent / "images" /
                (camera_name + extension)
                for extension in (".jpg", ".jpeg", ".png", ".bmp")
                if (frame_json.parent / "images" /
                    (camera_name + extension)).is_file()
            ]
            if len(fallback_images) == 1:
                image_path = fallback_images[0]
        if not image_path.is_file():
            raise FileNotFoundError(
                "{} image does not exist: {}".format(
                    camera_name, image_path))
        with Image.open(str(image_path)) as image:
            width, height = image.size
        camera_data[camera_name] = {
            "intrinsic": intrinsic,
            "camera_to_lidar": camera_to_lidar,
            "image_size": (width, height),
        }
    return camera_data


def occupancy_to_bev(occupancy, semantic, threshold):
    """Collapse each vertical voxel column to its highest occupied class."""
    if semantic:
        voxel_labels = occupancy.astype(np.int32, copy=False)
    else:
        if not np.issubdtype(occupancy.dtype, np.number):
            raise ValueError("Binary occupancy values must be numeric")
        if not np.all(np.isfinite(occupancy)):
            raise ValueError("Binary occupancy contains non-finite values")
        voxel_labels = (occupancy >= threshold).astype(np.int32)

    top_down = voxel_labels[:, :, ::-1] > 0
    occupied = top_down.any(axis=2)
    top_index = top_down.argmax(axis=2)
    z_index = occupancy.shape[2] - 1 - top_index
    bev = np.zeros(occupancy.shape[:2], dtype=np.int32)
    voxel_x, voxel_y = np.nonzero(occupied)
    bev[voxel_x, voxel_y] = voxel_labels[
        voxel_x, voxel_y, z_index[voxel_x, voxel_y]]
    return bev


def _make_color_map():
    color_map = ListedColormap(CLASS_COLORS.astype(np.float32) / 255.0)
    norm = BoundaryNorm(
        np.arange(-0.5, len(CLASS_COLORS) + 0.5, 1),
        color_map.N,
    )
    return color_map, norm


def _render_bev(
        axes, bev, point_cloud_range, forward_direction, color_map, norm):
    visible_bev = np.ma.masked_where(bev == 0, bev)
    forward = np.asarray(forward_direction[:2], dtype=np.float64)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Front camera has no horizontal viewing direction")
    forward /= forward_norm
    right = np.array([forward[1], -forward[0]])

    x_edges = np.linspace(
        point_cloud_range[0], point_cloud_range[3], bev.shape[0] + 1)
    y_edges = np.linspace(
        point_cloud_range[1], point_cloud_range[4], bev.shape[1] + 1)
    world_x, world_y = np.meshgrid(x_edges, y_edges, indexing="ij")
    horizontal = world_x * right[0] + world_y * right[1]
    longitudinal = world_x * forward[0] + world_y * forward[1]
    axes.pcolormesh(
        horizontal,
        longitudinal,
        visible_bev,
        shading="flat",
        cmap=color_map,
        norm=norm,
    )
    axes.set_xlim(horizontal.min(), horizontal.max())
    axes.set_ylim(longitudinal.min(), longitudinal.max())
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("Right (m)")
    axes.set_ylabel("Forward (m)")
    axes.set_title("Bird's-eye view")
    axes.grid(color="gray", alpha=0.25, linewidth=0.5)
    axes.scatter(
        [0], [0], marker="o", s=35, c="white",
        edgecolors="black", linewidths=0.8, zorder=3)
    axes.annotate(
        "front",
        xy=(0, 7),
        xytext=(0, 2),
        ha="center",
        color="black",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )


def _render_camera_view(
        axes, points, labels, voxel_size, camera, color_map, norm):
    """Project occupied voxel footprints into one camera's image plane."""
    lidar_to_camera = np.linalg.inv(camera["camera_to_lidar"])
    rotation = lidar_to_camera[:3, :3]
    translation = lidar_to_camera[:3, 3]
    camera_points = (
        points @ rotation.T + translation)
    depths = camera_points[:, 2]

    corner_signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1],
            [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1],
            [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float32,
    )
    corner_offsets = (
        corner_signs * np.asarray(voxel_size, dtype=np.float32) * 0.5)
    lidar_corners = points[:, np.newaxis, :] + corner_offsets
    camera_corners = lidar_corners @ rotation.T + translation
    corner_depths = camera_corners[:, :, 2]
    in_front = np.all(corner_depths > 1e-3, axis=1)
    projected_corners = camera_corners @ camera["intrinsic"].T
    corner_u = projected_corners[:, :, 0] / projected_corners[:, :, 2]
    corner_v = projected_corners[:, :, 1] / projected_corners[:, :, 2]
    u_min = corner_u.min(axis=1)
    u_max = corner_u.max(axis=1)
    v_min = corner_v.min(axis=1)
    v_max = corner_v.max(axis=1)

    width, height = camera["image_size"]
    valid = (
        in_front &
        (u_max > 0) & (u_min < width) &
        (v_max > 0) & (v_min < height)
    )

    axes.set_facecolor("#202020")
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(camera["name"], fontsize=9)
    axes.axis("off")

    if not np.any(valid):
        axes.text(
            0.5, 0.5, "No occupied voxels in view",
            color="white", ha="center", va="center",
            transform=axes.transAxes, fontsize=8)
        return

    valid_indices = np.flatnonzero(valid)
    order = np.argsort(depths[valid_indices])[::-1]
    valid_indices = valid_indices[order]
    polygons = [
        cv2.convexHull(
            np.column_stack((
                corner_u[index],
                corner_v[index],
            )).astype(np.float32)).reshape(-1, 2)
        for index in valid_indices
    ]
    collection = PolyCollection(
        polygons,
        edgecolors="none",
        linewidths=0,
        antialiased=False,
        cmap=color_map,
        norm=norm,
    )
    collection.set_array(labels[valid_indices])
    axes.add_collection(collection)


def visualize_prediction(
        path, threshold=0.5, output=None, frame_json=DEFAULT_FRAME_JSON):
    """Save a BEV and six calibrated camera-view projections to one PNG."""
    occupancy, semantic, point_cloud_range, class_names = load_prediction(path)
    points, labels, voxel_size = occupancy_to_points(
        occupancy, semantic, point_cloud_range, threshold)
    camera_data = load_camera_metadata(frame_json)
    bev = occupancy_to_bev(occupancy, semantic, threshold)

    prediction_path = Path(path).expanduser()
    if output is None:
        output = prediction_path.with_name(
            prediction_path.stem + "_views.png")
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    color_map, norm = _make_color_map()
    figure = plt.figure(figsize=(18, 9))
    grid = figure.add_gridspec(
        3,
        4,
        width_ratios=(1.15, 1.0, 1.0, 0.06),
        wspace=0.12,
        hspace=0.16,
    )
    bev_axes = figure.add_subplot(grid[:, 0])
    front_direction = camera_data["CAM_FRONT"]["camera_to_lidar"][:3, 2]
    _render_bev(
        bev_axes, bev, point_cloud_range,
        front_direction, color_map, norm)

    for index, camera_name in enumerate(CAMERA_ORDER):
        row, column = divmod(index, 2)
        axes = figure.add_subplot(grid[row, column + 1])
        camera = camera_data[camera_name]
        camera["name"] = camera_name
        _render_camera_view(
            axes, points, labels, voxel_size,
            camera, color_map, norm)

    colorbar_axes = figure.add_subplot(grid[:, 3])
    legend_labels = np.arange(len(class_names), dtype=np.int32)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=color_map),
        cax=colorbar_axes,
        ticks=legend_labels,
    )
    colorbar.set_ticklabels([
        class_names[label] if label < len(class_names)
        else "class {}".format(label)
        for label in legend_labels
    ])
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        "SurroundOcc occupancy views ({} occupied voxels)".format(
            len(points)),
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.035, right=0.97, top=0.92, bottom=0.05)
    figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("Saved occupancy visualization to {}".format(
        output_path.resolve()))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render a SurroundOcc prediction as a BEV and six camera views."))
    parser.add_argument(
        "prediction",
        nargs="?",
        default="result/occupancy_prediction.npz",
        help="Prediction archive (default: occupancy_prediction.npz).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Occupancy probability threshold for non-semantic predictions.")
    parser.add_argument(
        "--output",
        help=(
            "PNG path (default: alongside the prediction with a "
            "_views.png suffix)."))
    parser.add_argument(
        "--frame-json",
        default=str(DEFAULT_FRAME_JSON),
        help=(
            "Frame JSON containing camera calibration and image paths "
            "(default: real_data/frame.json)."))
    args = parser.parse_args()
    visualize_prediction(
        args.prediction,
        threshold=args.threshold,
        output=args.output,
        frame_json=args.frame_json,
    )


if __name__ == "__main__":
    main()

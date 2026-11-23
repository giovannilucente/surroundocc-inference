# SurroundOcc Real-Camera Inference

This repository provides a Docker environment and small Python tools for
running SurroundOcc inference on one synchronized frame from six cameras. It
includes the customized SurroundOcc project as a Git submodule, a runner that
loads camera images and calibration from JSON, and a script that visualizes
the resulting occupancy prediction.

This is an inference wrapper, not a ROS implementation or a training setup.
It does not include camera data or a trained occupancy checkpoint.

## Repository layout

```text
.
├── Dockerfile
├── SurroundOcc/                 # Git submodule
├── surroundocc_run.py           # Run inference on one frame
├── visualize_occupancy.py       # BEV and six camera-view occupancy render
├── real_data/                   # Local input data; ignored by Git
└── result/                      # Local outputs; ignored by Git
```

## Requirements

- Linux with Docker and the NVIDIA Container Toolkit installed.
- An NVIDIA GPU and a compatible host NVIDIA driver. The image uses CUDA
  11.1.1; the host CUDA toolkit does not need to be installed separately.
- A trained SurroundOcc **occupancy** checkpoint compatible with the supplied
  inference config.
- Six synchronized images and their camera calibration.

The Docker image uses Python 3.7, PyTorch 1.10.1 with CUDA 11.1, MMCV 1.4.0,
MMDetection 2.14.0, MMSegmentation 0.14.1, and MMDetection3D v0.17.1. No
Conda environment is used.

The default CUDA architecture is 8.6. For a V100, build with
`--build-arg TORCH_CUDA_ARCH_LIST=7.0`. Set this to match the GPU on which the
image will run.

## Get the source

Clone this repository together with its SurroundOcc submodule:

```bash
git clone --recurse-submodules \
  https://github.com/giovannilucente/surroundocc-inference.git
cd surroundocc-inference
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build the Docker image

Run this from the repository root (the directory containing `Dockerfile`):

```bash
docker build -f Dockerfile -t surroundocc:cuda11.1 .
```

The final `.` is important: it makes the outer repository the build context,
so Docker can copy the Chamfer Distance extension from the `SurroundOcc`
submodule.

For a V100:

```bash
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST=7.0 \
  -f Dockerfile \
  -t surroundocc:cuda11.1 .
```

## Prepare a camera frame

The inference runner expects one JSON file describing exactly these six
cameras:

```text
CAM_FRONT
CAM_FRONT_RIGHT
CAM_BACK_RIGHT
CAM_BACK
CAM_BACK_LEFT
CAM_FRONT_LEFT
```

By default, the runner looks for `real_data/frames.json`, then
`real_data/frame.json`. The JSON may contain a `frame_id` and must contain a
`cameras` object. Each camera entry must provide:

- `image`: image path, absolute or relative to the JSON file.
- `intrinsic`: a 3x3 camera intrinsic matrix.
- `camera_to_lidar`: a 4x4 transform from that camera's coordinates into the
  common LiDAR/reference frame.

All six images must be from the same time and have matching dimensions. For
example, place a frame file at `real_data/frame.json` and its images under
`real_data/images/`. The JSON image paths should refer to those files.

The frame JSON, camera images, and generated results are intentionally
excluded by `.gitignore`. Supply your own data locally; do not commit private
or non-redistributable camera data.

## Start the container

Run this from the repository root on the host:

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  surroundocc:cuda11.1 bash
```

Inside the container, check that PyTorch can see the GPU:

```bash
python3.7 -c "import torch; print(torch.cuda.is_available())"
```

This should print `True`. If it does not, check the host NVIDIA driver and
NVIDIA Container Toolkit configuration.

## Run inference

Put a trained occupancy checkpoint somewhere under the mounted workspace,
for example at `checkpoints/surroundocc_occ.pth`. The Dockerfile does not
download an occupancy checkpoint; the backbone pretraining weights are not a
replacement for a trained occupancy checkpoint.

From inside the container, run:

```bash
python3.7 surroundocc_run.py \
  --checkpoint /workspace/checkpoints/surroundocc_occ.pth \
  --frame-json /workspace/real_data/frame.json \
  --output /workspace/result/occupancy_prediction.npz
```

The default config is
`SurroundOcc/projects/configs/surroundocc/surroundocc_inference.py`. Override
it with `--config /workspace/path/to/config.py` if needed.

The output is a compressed NumPy archive containing the dense occupancy
prediction, confidence, semantic flag, point-cloud range, and class names.
The `result/` directory is ignored by Git.

## Visualize the prediction

In the same container, run:

```bash
python3.7 visualize_occupancy.py \
  result/occupancy_prediction.npz \
  --frame-json real_data/frame.json
```

This writes `result/occupancy_prediction_views.png` by default. The image
contains a bird's-eye view and occupancy projections for all six cameras,
with a legend for the model's classes. The camera panels are projections of
the occupancy voxels using the calibration in the frame JSON; they are not
overlays on the original camera images.

To select other files:

```bash
python3.7 visualize_occupancy.py path/to/prediction.npz \
  --frame-json path/to/frame.json \
  --output path/to/visualization.png
```

## Using the inference API

`SurroundOcc/tools/infer.py` also exposes reusable Python functions and a
`SurroundOccRunner` class. It loads the model and fixed calibration once,
then accepts synchronized images as paths or BGR NumPy arrays. This API can
be imported by another application; this repository does not include a ROS
node.

## Data, checkpoints, and licensing

Camera images, frame calibration JSON, checkpoints, and generated outputs
are not included in this repository. Confirm that you have permission to use
and redistribute any data or weights before sharing them. SurroundOcc is
included as a submodule from this project's fork; its license and notices
remain in that repository.

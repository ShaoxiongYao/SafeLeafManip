# Safe Leaf Manipulation for Accurate Shape and Pose Estimation of Occluded Fruits

**Shaoxiong Yao<sup>1†</sup>, Sicong Pan<sup>2†</sup>, Maren Bennewitz<sup>2</sup>, Kris Hauser<sup>1</sup>**  
<sup>1</sup>University of Illinois Urbana-Champaign, USA  
<sup>2</sup>Humanoid Robots Lab, University of Bonn, Germany  
<sup>†</sup>Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2409.17389-b31b1b.svg)](https://arxiv.org/abs/2409.17389)  [![YouTube Video](https://img.shields.io/badge/Video-Youtube-red?logo=youtube)](https://youtu.be/bHzx8Zcsfoo) [![Project Website](https://img.shields.io/badge/Website-Project-blue?logo=githubpages&logoColor=white)](https://shaoxiongyao.github.io/lmap-ssc/)


<p align="center">
   <img width="600" alt="star_figure_v3" src="https://github.com/user-attachments/assets/72f90d31-9b38-4290-8b72-eb02ac3ff3ea" />
</p>

## Overview

This repository contains the official code implementation for our ICRA 2025 paper.

We address the challenge of **monitoring occluded fruits** in agricultural environments. The provided functions enable robots to **autonomously plan grasp-and-pull actions** that enhance fruit visibility while minimizing leaf damage.

## Features

- **Plant segmentation with GroundedSAM**: Segment RGB-D images into point clouds of leaves, branches, and fruits.
- **Scene-consistent shape completion**: Reconstruct full branch and fruit shapes from partial point cloud data.
- **Grasp and pull action planning**: Sample candidate grasp points and corresponding pull directions on leaves.
- **Leaf deformation simulation**: Generate an embedded deformation graph from the completed plant model and simulate action outcomes.
- **Visibility prediction via OctoMap**: Predict fruit visibility changes resulting from leaf manipulation.

## Installation

### Dependencies

Minimum required Python packages:

- `numpy`
- `torch`
- `pypose`
- `open3d`
- `dacite`
- `octomap-python`

### Conda Environment 

To set up a conda environment with all dependencies, use the provided `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate ssc_lmap
```

### Installing `octomap-python`

To install `octomap-python`, make sure to update your C++ compile flags before using pip:

```bash
export CXXFLAGS="-std=c++11"
pip install octomap-python
```

The following modules are optional to run plant semantic segmentation and robot action validation:

<details>
<summary>Prepare GroundedSAM Environment for Plant Segmentation</summary>


> **Note:** This setup is only required if you want to re-run the segmentation step. For shape completion and deformation simulation, pre-segmented masks are already provided in the demo data.

1. **Clone Required Repositories**

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
git clone https://github.com/facebookresearch/segment-anything.git
```

2. **Install Dependencies**

Follow the [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) instructions to install both modules:

```bash
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
```

You're now ready to run plant segmentation using GroundedSAM.
</details>

<details>
<summary>Setup UR5 Simulator in Docker </summary>

1. **Install Docker on Linux**  
   Follow instructions at: https://docs.docker.com/engine/install/

2. **Pull the URSim Docker Image**  
   ```bash
   docker pull universalrobots/ursim_e-series
   ```

3. **Start the URSim Container**  
   ```bash
   docker run --rm -it --net=host universalrobots/ursim_e-series
   ```

4. **Get the Robot IP Address**  
   When URSim starts, note the IP address shown when you start the container.  
   Update your control scripts to use this IP.

5. **Turns on Simulator and Enable Remote Control in URSim**  
   Inside the URSim GUI:
   - Press the **power button** (bottom-left), then press **Start**
   - Go to **Settings → System → Remote Control**
   - Click **Enable**

You're now ready to run your control pipeline.
</details>


## Use Cases

We provide two demo scripts to illustrate the functionality of the framework.

### Step 1: Download Test Data

Download the `test_obj_100` folder from the following link:

🔗 [https://uofi.box.com/s/hu77m4usywjo0segpzuoxuvyw1gohl48](https://uofi.box.com/s/hu77m4usywjo0segpzuoxuvyw1gohl48)

Place the downloaded `test_obj_100` folder in your desired working directory.

Set the `DATA_DIR` environment variable to the path of the `test_obj_100` folder:

```bash
DATA_DIR="path/to/test_obj_100"
```


### 1. Run Shape Completion Demo

This script performs **scene-consistent shape completion**, taking segmented plant point clouds as input and generating completed shapes for branches and fruits.

```bash
python scripts/plant_shape_completion.py \
    --segment_dir "${DATA_DIR}/segmented" \
    --shape_complete_dir "${DATA_DIR}/completed" \
    --trans_params_fn "${DATA_DIR}/raw/extrinsic.json" \
    --shape_complete_config configs/shape_complete.yaml
```


### 2. Run Deformation Simulation Demo

This script samples several grasp-and-pull actions around the robot and simulates their effects using an embedded deformation model. It outputs the simulated ARAP energy and the number of visible fruit points after each action.

```bash
python scripts/plant_simulate_action.py \
    --segment_dir "${DATA_DIR}/segmented" \
    --shape_complete_dir "${DATA_DIR}/completed" \
    --trans_params_fn "${DATA_DIR}/raw/extrinsic.json" \
    --simulate_action_config configs/simulate_action.yaml \
    --sim_out_dir "${DATA_DIR}/sim_out"
```

## References
```
TODO: add refences when ICRA proceeding published.
```

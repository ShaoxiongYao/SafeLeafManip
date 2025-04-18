# SafeLeafManip

## Installation
You can create conda environment using `environment.yaml`.

### Prepare GroundedSAM Environment for Plant Segmentation

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

### Run Complete Pipeline with UR5 Simulation (URSim)

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
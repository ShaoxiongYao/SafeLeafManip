# SafeLeafManip


---

## Run Complete Pipeline with UR5 Simulation (URSim)

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
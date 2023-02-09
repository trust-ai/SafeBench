<!--
 * @Author: 
 * @Email: 
 * @Date: 2023-01-25 19:36:50
 * @LastEditTime: 2023-02-04 17:43:12
 * @Description: 
-->

# SafeBench

## Installation
1. Setup conda environment
```
conda create -n safebench python=3.7
conda activate safebench
```

2. Clone this git repo in an appropriate folder
```
git clone git@github.com:trust-ai/SafeBench_v2.git
```

3. Enter the repo root folder and install the packages:
```
pip install -r requirements.txt
pip install -e .
```

4. Download [CARLA_0.9.13](https://github.com/carla-simulator/carla/releases), extract it to your folder.

5. Run `sudo apt install libomp5` as per this [git issue](https://github.com/carla-simulator/carla/issues/4498).

6. Add the python API of CARLA to the ```PYTHONPATH``` environment variable. You can add the following commands to your `~/.bashrc`:
```
export CARLA_ROOT={path/to/your/carla}
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

## Usage

### Desktop Users

Enter the CARLA root folder, launch the CARLA server and run our platform with
```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch SafeBench in another terminal
python scripts/run.py --agent_cfg=dummy.yaml --scenario_cfg=example.yaml
```

### Remote Server Users
Enter the CARLA root folder, launch the CARLA server with headless mode, and run our platform with
```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Launch SafeBench in another terminal
SDL_VIDEODRIVER="dummy" python scripts/run.py --agent_cfg=dummy.yaml --scenario_cfg=example.yaml
```

(Optional) You can also visualize the pygame window using [TurboVNC](https://sourceforge.net/projects/turbovnc/files/).
First, launch CARLA with headless mode, and run our platform on a virtual display.
```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Run a remote VNC-Xserver. This will create a virtual display "8".
/opt/TurboVNC/bin/vncserver :8

# Launch SafeBench on the virtual display
DISPLAY=:8 python scripts/run.py --agent_cfg=dummy.yaml --scenario_cfg=example.yaml
```

You can use the TurboVNC client on your local machine to connect to the virtual display.
```
/opt/TurboVNC/bin/vncviewer -via user@host localhost:n
```
where `user@host` is your remote server, and n is the display port specified when you started the VNC server on the remote server ("8" in our example).
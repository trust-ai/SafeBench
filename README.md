<!--
 * @Author: 
 * @Email: 
 * @Date: 2023-01-25 19:36:50
 * @LastEditTime: 2023-02-04 17:32:25
 * @Description: 
-->

# SafeBench

## Installation
1. Setup conda environment
```
$ conda create -n safebench python=3.7
$ conda activate safebench
```

2. Clone this git repo in an appropriate folder
```
$ git clone git@github.com:trust-ai/SafeBench_v2.git
```

3. Enter the repo root folder and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
```

4. Download [CARLA_0.9.13](https://github.com/carla-simulator/carla/releases), extract it to your folder, and add the python API of CARLA to the ```PYTHONPATH``` environment variable. You can add the following commands to your `~/.bashrc`:
```
export CARLA_ROOT={path/to/your/carla}
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

## Usage
1. Enter the CARLA root folder, launch the CARLA server and run our platform with
```
$ cd {path/to/your/carla}
$ ./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
$ python scripts/run.py
```
If you want to run Carla with headless mode, please use the following commands:
```
$ ./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000
$ SDL_VIDEODRIVER="dummy" python scripts/run.py
```

# SafeBench

## Installation
1. Setup conda environment
```
$ conda create -n env_name python=3.7
$ conda activate env_name
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

4. Download [CARLA_0.9.11](https://github.com/carla-simulator/carla/releases/tag/0.9.11), extract it to some folder, and add CARLA to ```PYTHONPATH``` environment variable. You can add the following environment variables to your `~/.bashrc`:
```
export CARLA_ROOT=/home/zuxin/carla-0.9.11 # replace this with your own dir
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

## Usage
1. Enter the CARLA root folder and launch the CARLA server by:
```
$ ./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
```
You can use ```Alt+F1``` to get back your mouse control.

Or you can run in on a remote server with non-display mode by:
```
$ DISPLAY= ./CarlaUE4.sh -prefernvidia -opengl -carla-port=2000
```

2. Run the test file:
```
$ python test.py
```
See details of ```test.py``` about how to use the CARLA gym wrapper.

Note that if you are running on a remote server via ssh, you have to first create a virtual screen and specify the DISPLAY number:
```
Xvfb :99 -screen 0 1024x768x16 &
```
Then export the DISPLAY env variable:
```
export DISPLAY=:99
```
Then run `test.py` as usual.



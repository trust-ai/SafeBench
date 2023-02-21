<!--
 * @Author: Wenhao Ding
 * @Email: wenhaod@andrew.cmu.edu
 * @Date: 2021-07-18 21:46:37
 * @LastEditTime: 2023-02-21 13:35:50
 * @Description: 
-->

# Installation

We provide a detailed instruction of how to install Safebench. The installation does not require docker or ROS.

## Step 1. Setup Safebench

We recommand using anaconda for creating a clean environment.
```
conda create -n safebench python=3.8
conda activate safebench
```

Then, clone the code from github in an appropriate folder with
```
git clone git@github.com:trust-ai/SafeBench_v2.git
```

Enter the folder of safebench and install some necessary packages
```
cd SafeBench_v2
pip install -r requirements.txt
pip install -e .
```

## Step 2. Setup Carla

Download our built [CARLA_0.9.13](https://drive.google.com/file/d/1Ta5qtEIrOnpsToQfJ-j0cdRiF7xCbLM3/view?usp=share_link) and extract it to your folder with
```
mkdir carla && cd carla
tar -zxvf CARLA_0.9.13-2-g0c41f167c-dirty.tar.gz
```

Add the Python API of Carla to the ```PYTHONPATH``` environment variable. You can add the following commands to your `~/.bashrc`:
```
export CARLA_ROOT={path/to/your/carla}
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

## Potential Issue

Run `sudo apt install libomp5` as per this [git issue](https://github.com/carla-simulator/carla/issues/4498).


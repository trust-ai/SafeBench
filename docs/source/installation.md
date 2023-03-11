<!--
 * @Date: 2021-07-18 21:46:37
 * @LastEditTime: 2023-03-11 16:59:26
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
git clone git@github.com:trust-ai/SafeBench.git
```

Enter the folder of safebench and install some necessary packages
```
cd SafeBench
pip install -r requirements.txt
pip install -e .
```

## Step 2. Setup Carla

Download our built [CARLA_0.9.13](https://drive.google.com/file/d/1A4z3RKXqVYpOmsEZkPBV1Pbw3B8aeSMp/view?usp=sharing) and extract it to your folder with
```
mkdir carla && cd carla
tar -zxvf CARLA_0.9.13_safebench_2.tar.gz
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


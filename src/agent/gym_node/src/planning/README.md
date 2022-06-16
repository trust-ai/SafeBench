## Planning repo usage
==================================
## Environment Setup
### System Requirements
- Tested in Ubuntu 20.04, should be fine with Ubuntu 18.04
- I would recommend to use [Anaconda3](https://docs.anaconda.com/anaconda/install/) for python env management

### Env Setup
cd to the red_team root folder and
```
pip install -e .
```

When you run the training jobs, conda or pip install the packages if anything is missing.

### TODO (for Weijie):
* ~~clean structure~~
* Try this repo and train a ppo agent with 4-states obs so that you can get familiar with this repo and some basic tools.
* Modify the `env_wrapper.py` to incorporate prediction embeddings to the state space. Use a different `OBS_TYPE` for any new observations you added.
* Train another ppo agent with the new observation type, compare the performance with `OBS_TYPE=0` (the 4-states one).


### How to Run
Simply run
```
python run.py -p ppo
```
where `-p` is the policy name. More parameters could be found in `config` folder and in `run.py` and in `carla_runner.py`.

To evaluate a trained model, run:
```
python run.py -m eval -d /model_dir
```
Some pretrained models could be found [here]
(https://drive.google.com/file/d/19SkluwA7qXlSCPGHjttFf91In2z_bouA/view?usp=sharing).

### Choose different prediction modules
For now, there are 4 different obs types we can give to the RL model. It can be chosen in `env_wrapper.py`.
- `OBS_TYPE = 0`: Directly use *lateral_dis, -delta_yaw, speed, self.vehicle_front* from `obs` as the input of RL module.
- `OBS_TYPE = 1`: Use **Implicit Affordance** algorithm from **End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances** to process front camera view image.
- `OBS_TYPE = 2`: Use **CoverNet algorithm** from **CoverNet: Multimodal Behavior Prediction using Trajectory Sets** to process bird-eye view image and other state parameters *speed, acc, heading_change_rate*. The output is the probilities of trajectories in a trajectory set (range from 0 to 1).
- `OBS_TYPE = 3`: Use agent motion prediction algorithm from **L5Kit**.
- To switch `OBS_TYPE`, simply change the parameter of `CFG` in `env_wrapper.py`.

### Structure
The structure of this repo is as follows:
```
planning
├── safe_rl  # core package folder
│   ├── policy # safe model-free RL methods
│   ├── ├── model # stores the actor critic model architecture
│   ├── ├── policy_name # RL algorithms implementation
│   ├── util # logger and pytorch utils
│   ├── worker # collect data from the environment
├── config  # stores the environment configurations.
├── script  # stores the Webots-based gym style simulation environment.
├── env_wrapper.py # bridge between the env, perception, prediction, with the RL planner
├── carla_runner.py # core module to connect policy and worker
├── run.py # launch a single experiment
├── plot.py # plot the training results
├── data # stores experiment results
```

## Check experiment results

You may either use `tensorboard` or `plot.py` to monitor the results. All the experiment results are stored in the `data` folder with corresponding experiment name.

For example:
```
tensorboard --logdir data/experiment_folder
python plot.py data/experiment_folder -y EpRet EpCost
```

#### Tips
Tensorboard could be used to monitor the training process, simply run:
```
tensorboard --logdir /dir/to/the/result/folder



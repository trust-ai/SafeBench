<!--
 * @Author: Wenhao Ding
 * @Email: wenhaod@andrew.cmu.edu
 * @Date: 2021-07-18 21:46:37
 * @LastEditTime: 2021-07-23 14:16:30
 * @Description: 
-->

# Quick start

* __[Environment Settings](#environment-settings)__  
    * [Docker Installation](#docker-installation)  
	* [Image Download](#image-download)  
	* [Source Code Download](#source-code-download)  
	* [Run the Docker Container](#run-the-docker-container)  
	* [Run the Platform](#run-the-platform)  

---

Firstly, make sure you already install the NVIDIA driver on your mechine. All environment settings are store in a docker image, please follow the instructions below to install all things.

## Environment Settings

### Docker Installation
1. Install Docker by following [this link](https://docs.docker.com/engine/install/ubuntu/).
2. Install NVIDIA-Docker2 by following [this link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### Image Download
Pull the [Docker image](https://hub.docker.com/r/gilgameshd/platform): `docker pull gilgameshd/platform`

### Source Code Download
Download the source code from this repo: 
`git clone https://github.com/trust-ai/Evaluation-Platform.git`

### Run the Docker Container
The command of running the container is in `run_docker.sh`, you just need to run this script. After running it, a window of the Carla Simulator will show up.

### Run the Platform
1. Open a new terminal window and run the script `run_bash.sh` to access the bash of container.
2. Change directory: `cd Evaluation`
3. Complile all files with ROS tools: `catkin_make`
4. Set up environment: `. ./devel/setup.bash`
5. launch platform: `roslaunch manager manager.launch`

Finally, you should be able to see that the Carla window changes the map and spawns an ego vehicle. Another window of pygame will also show up for controlling the ego vehicle.

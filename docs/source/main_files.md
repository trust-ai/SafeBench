<!--
 * @Author: Shuai Wang
 * @Email: shuaiwa2@andrew.cmu.edu
 * @Date: 2022-11-09 11:22:05
 * @LastEditTime: 2023-02-21 13:49:45
 * @Description: 
-->

# Functions and API
In this section we summarize some important files and functions, for developer's reference


## Carla Runner

### /src/manager/src/manager.py
This file is responsible for creating manager node, scenario node and gym node, parameters like scenario_id, route_id, obs_type, etc are coming from SafeBench/src/manager/launch/manager.launch
* \__main\__(): initialize manager node
* create_agent_and_scene():
    * roslaunch create scenario node, launch file is SafeBench/src/manager/launch/create_scenario.launch, arguments 
  like route_id, scenario_id come from manager node class
    * roslaunch create gym node, arguments like policy, obs_type, mode come from manager node class variables

## Agent Module

## Gym module

### /src/agent/gym_node/src/planning/run.py
This file is the entrance of the gym node, main work is creating the carla runner
object and running with the chosen mode

### /src/agent/gym_node/src/planning/carla_runner.py
This file defines carla runner class, control the main training and evaluation process
* \__init\__(): initialize self.env with the carla_env defined in SafeBench/src/agent/gym_node/src/planning/env_wrapper.py
* eval(): get the observations from self.env, snd to policy as the input to get the actions,
then send the actions back to self.env

### /src/agent/gym_node/src/planning/env_wrapper.py
* carla_env(): create carla environment defined in SafeBench/src/agent/gym_node/src/gym_carla/carla_env.py
* \__init\__(): initialize self._env by carla_env()
* step(): called by SafeBench/src/agent/gym_node/src/planning/carla_runner.py, get actions, apply actions
to self._env() and get obs from self._env

### /src/agent/gym_node/src/gym_carla/carla_env.py
The real carla environment
* \__init\__(): get the carla world
* initialize(): get the ego clusters, collision sensors, camera sensors, lidar sensors and all the actors polygons
* step(): called by SafeBench/src/agent/gym_node/src/planning/env_wrapper.py, get actions and apply actions to
all the ego cluster, return observations get from carla world
* _get_obs(): get the observations from the world

## Scenario Module

### /pkgs/scenario_runner/scenario_runner.py
This file is the highest level file for scenario runner, mainly responsible for initializing,
running and stopping a single or a list of scenarios. Scenarios can run in parallel mode in multiple threads
* _run_route(): in this platform the scenario is running in route mode, this function is the start of the scenario runner
* _load_world(): function is used to load the world in Carla environment by the town name given by the arguments
* _load_run_scenario(): function is used to load and run the route scenario, the scenario loading
and running procedure is controlled by scenario manager created

### /pkgs/scenario_runner/srunner/scenariomanager/scenario_manager_dynamic.py
This file contains the definition of the scenario manager, which is the main controller of the scenario running process.
Scenario manager is responsible for initializing, running and destroying the scenarios
* load_scenario(): get the scenario on the current selected route and store in scenario_list
* run_scenario(): function to initialize and run all the scenarios in the scenario list, 
call _init_scenarios() function and _get_update() function
* _init_scenarios(): iterate through scenario_list and initialize all the scenarios needed
* _get_update(): while the scenario is in running status, in each timestamp, update the behavior
of behaviors of all the actors in each scenario iteratively

### /pkgs/scenario_runner/srunner/scenario_dynamic/basic_scenario_dynamic.py
The file contains the base class for all the other types of scenarios, there are several functions need to be implemented
by the scenarios inherited from this base class, here lists two most important ones
* initialize_actors(): function is used to initialize all the actors needed in this scenario
* update_behavior(): function is used to update the actors' behaviors in each timestamp

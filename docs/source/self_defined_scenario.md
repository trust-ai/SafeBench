<!--
 * @Author: 
 * @Email: 
 * @Date: 2023-02-20 20:13:41
 * @LastEditTime: 2023-02-21 00:48:06
 * @Description: 
-->

# How to Create Your Own Scenario


## Step1: Define Scenario Class
All the scenario class need to inherit BasicScenarioDynamic base class defined in  pkgs/scenario_runner/srunner/scenario_dynamic/basic_scenario_dynamic.py

## Step2: Implement Functions
There are mainly two functions need to implement
1. initialize_actors(): initialize and spawn all actors
2. update_behavior(): used for manager to update each actor’s behavior in each timestamp

## Additional Tools
pkgs/scenario_runner/srunner/AdditionTools/scenario_operation.py has many functions can be used to initialize actors and apply particular control commands to actors



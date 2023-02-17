# Create routes
Run the following command to create route waypoints.
```
python create_waypoints.py
optional arguments:
  --map Name of the map
  --save_dir Output directory
  --scenario Scenario index
  --road {intersection,straight} Create route based on the intersection or straight road
  --multi_rotation      This will create multiple routes that is symetry. For example for intersection, this will create 4 routes in total with ego vehicle coming from different direction
```

# Create Scenarios
Run the following command to create scenario trigger point and other actor spawn point.
```
python create_scenario.py
optional arguments:
  --map Name of the map
  --save_dir Output directory
  --scenario Scenario index
  --road {intersection,straight} Create route based on the intersection or straight road
  --multi_rotation      This will create multiple routes that is symetry. For example for intersection, this will create 4 routes in total with ego vehicle coming from different direction
```

# Visualize Routes
To visualize the route, run command
```
python visualize_waypoints.py.py
optional arguments:
  --map Name of the map
  --save_dir Output directory
  --scenario Scenario index
```

# Visualize Scenarios
To visualize the route, run command
```
python visualize_scenario.py
optional arguments:
  --map Name of the map
  --save_dir Output directory
  --scenario Scenario index
```
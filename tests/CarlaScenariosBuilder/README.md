# Create routes
To create route waypoints, run the following command.
```
python create_routes.py

optional arguments:
  --map                          Name of the map
  --save_dir                     Output directory
  --scenario                     Scenario index
  --road {intersection,straight} Create routes based on a intersection or a straight road.
  --multi_rotation               Create multiple symmetrical routes. 
                                 - When creating routes within the intersection, the code will generate four routes, 
                                   each rotated 90 degrees around the intersection center. 
                                 - When creating routes alone the straight road, the code will generate two routes, 
                                   each rotated 180 degrees around the road center.
```

# Create Scenarios
To create the scenario trigger point and spawn points for other actors, run the following command.
```
python create_scenario.py

optional arguments:
  --map                          Name of the map
  --save_dir                     Output directory
  --scenario                     Scenario index
  --road {intersection,straight} Create routes based on a intersection or a straight road.
  --multi_rotation               Create multiple symmetrical routes. 
                                 - When creating routes within the intersection, the code will generate four routes, 
                                   each rotated 90 degrees around the intersection center. 
                                 - When creating routes alone the straight road, the code will generate two routes, 
                                   each rotated 180 degrees around the road center.
```

# Visualize Routes
To visualize the routes, run the following command.
```
python visualize_routes.py

optional arguments:
  --map                          Name of the map
  --save_dir                     Output directory
  --scenario                     Scenario index
```

# Visualize Scenarios
To visualize trigger point and spawn points for other actors of a scenario, run the following command
```
python visualize_scenarios.py

optional arguments:
  --map                          Name of the map
  --save_dir                     Output directory
  --scenario                     Scenario index
```
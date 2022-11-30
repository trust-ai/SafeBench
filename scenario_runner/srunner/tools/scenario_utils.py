import math


def calculate_distance_transforms(transform_1, transform_2):
    distance_x = (transform_1.location.x - transform_2.location.x) ** 2
    distance_y = (transform_1.location.y - transform_2.location.y) ** 2
    return math.sqrt(distance_x + distance_y)

def calculate_distance_locations(location_1, location_2):
    distance_x = (location_1.x - location_2.x) ** 2
    distance_y = (location_1.y - location_2.y) ** 2
    return math.sqrt(distance_x + distance_y)


import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add patch to the stopsign')
    # take the user image input
    image0 = cv2.imread('/home/ubuntu/carla/SafeBench/safebench/scenario/scenario_data/template_od/stopsign.jpg')


    # keep the same location with different size:
    circle_center = [310, 310]
    color_bgr = [160, 32, 240]
    for size in range(0, 101, 25):
        image1 = cv2.circle(image0, circle_center, size, color_bgr, -1)
        filename = '/home/ubuntu/carla/SafeBench/safebench/scenario/scenario_data/template_od/stopsign_patchsize_' + str(size) + '.jpg'
        cv2.imwrite(filename, image1)

    # patch_loc = np.array([310, 310]) + np.random.randint(0,50)
    size = 100
    for center_x in range(210, 411, 50):
        for center_y in range(210, 411, 50):
            image1 = cv2.circle(image0, [center_x, center_y], size, color_bgr, -1)
            filename = '/home/ubuntu/carla/SafeBench/safebench/scenario/scenario_data/template_od/stopsign_patchcenter_' + str(center_x) + '_' + str(center_y) + '.jpg'
            cv2.imwrite(filename, image1)
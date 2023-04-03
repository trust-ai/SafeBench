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
    # plot the original figure
    cv2.imshow('Input color image', image0)
    cv2.waitKey(0)
    patch_loc = np.array([310, 310]) + np.random.randint(0,50)
    image1 = cv2.circle(image0, [patch_loc[0],patch_loc[1]], 100, [160, 32, 240], -1)
    cv2.imshow('patched image', image1)
    cv2.waitKey(0)
    filename = '/home/ubuntu/carla/SafeBench/safebench/scenario/scenario_data/template_od/stopsign_patch.jpg'
    cv2.imwrite(filename, image1)
    cv2.waitKey(0)
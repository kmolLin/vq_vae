from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import cv2

if __name__ == "__main__":
    a = os.listdir("image_data")
    cc = 0
    for i in a:
        if i in 
        path = f"image_data/{i}/*.jpeg"
        for image_name in glob.glob(path)
            cv2.imread(f"{image_name}")
            cv2.imwrite(f"")
'''
Please do NOT make any changes to this file.
'''

import cv2
from typing import Dict, List
import json
import os
import numpy as np
import argparse
import zipfile


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()



def check_submission(py_file, check_list=['cv2.imshow(', 'cv2.imwrite(', 'cv2.imread(', 'open(']):
    res = True
    with open(py_file, 'r') as f:
        lines = f.readlines()
    for nline, line in enumerate(lines):
        for string in check_list:
            if line.find(string) != -1:
                print('You submitted code (in line %d) cannot have %s (Even if it is commented). Please remove that and zip again.' % (nline + 1, string[:-1]))
                res = False
    return res

def files2zip(files: list, zip_file_name: str):
    res = True
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            path, name = os.path.split(file)
            if os.path.exists(file):
                if name == 'UB_Face.py':
                    if not check_submission(file):
                        print('Zipping error!')
                        res = False
                zf.write(file, arcname=name)
            else:
                print('Zipping error! Your submission must have file %s, even if you does not change that.' % name)
                res = False
    return res

def parse_args():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument("--ubit", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    file_list = ['task1.py', 'task2.py', 'result_task1.json', 'result_task1_val.json', 'task2_overlap.txt', 'task2_result.png']
    res = files2zip(file_list, 'submission_' + args.ubit + '.zip')
    if not res:
        print('Zipping failed.')
    else:
        print('Zipping succeed.')
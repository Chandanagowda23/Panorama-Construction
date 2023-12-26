
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import array as arr
import utils
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Project.")
    parser.add_argument(
        "--input_path", type=str, default="data/images_panaroma",
        help="path to images for panaroma construction")
    parser.add_argument(
        "--output_overlap", type=str, default="./task2_overlap.txt",
        help="path to the overlap result")
    parser.add_argument(
        "--output_panaroma", type=str, default="./task2_result.png",
        help="path to final panaroma image ")

    args = parser.parse_args()
    return args

def stitch(inp_path, imgmark, N=4, savepath=''): 
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    
    imgs = []
    imgpath = [f'{inp_path}/{imgmark}_{n}.png' for n in range(1,N+1)]
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)

   
    "Start you code here"
    images = copy.deepcopy(imgs)
    overlap_array_output = overlap_array(images)
    for i in range(len(images)):
        i = 0
        best_score = float('-inf')
        best_pair = None
        for j in range(i + 1, len(images)):
            image1 = images[i]
            image2 = images[j]

            (keypoints1, features1) = extract_keypoint_feature_function(image1)
            (keypoints2, features2) = extract_keypoint_feature_function(image2)
            matched_features1 = matching_featues(features1, features2)
            
            overlap_percentage1 = len(matched_features1) / min(len(features1),len(features2))

            (Homography_Mat, status1) = finding_homography(keypoints1, keypoints2, matched_features1, 9)
            score1 = np.sum(status1)

            matched_features2 = matching_featues(features2, features1)
            (Homography_Mat_reverse, status2) = finding_homography(keypoints2, keypoints1, matched_features2,9)
            score2 = np.sum(status2)
            overlap_percentage2 = len(matched_features2) / min(len(features1), len(features2))

            if max(score1, score2) > best_score:
                if (score1 > score2) and (overlap_percentage1 >0.20):
                    best_score = score1
                    best_pair = (i, j, Homography_Mat)
                    stitch_img = stitching_image(images[j], images[i], Homography_Mat)
                    a = j
                    
                elif (score1 < score2) and (overlap_percentage2 >0.20):
                    best_score = score2
                    best_pair = (j, i, Homography_Mat_reverse)
                    stitch_img = stitching_image(images[i], images[j], Homography_Mat_reverse)
                    a = j
            else:
                pass       
        if i < j:
            stitch_img = black_padding_removal(stitch_img)         
            if len(images) > 2:
                images.pop(i)
                images.pop(a-1)
                images.insert(0, stitch_img)
        else:
            stitch_img = black_padding_removal(stitch_img)         
            if len(images) > 2:
                images.pop(a)
                images.pop(i-1)
                images.insert(0, stitch_img)

    cv2.imwrite('task2_result.png',stitch_img)   
    return overlap_array_output 
    
       
def extract_keypoint_feature_function(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shift_function = cv2.SIFT_create()
    (keypoints, features) = shift_function.detectAndCompute(image, None)
    return (keypoints, features)

def matching_featues(features1, features2):
    matched_lst = []
    for i, j in enumerate(features1):
        distance_between = np.linalg.norm(features2 - j, axis=1)
        sorted_val = np.argsort(distance_between)

        if distance_between[sorted_val[0]] < 0.75 * distance_between[sorted_val[1]]:
            matched_lst.append((i, sorted_val[0], distance_between[sorted_val[0]]))

    return matched_lst

def finding_homography(keypoints1, keypoints2, matched_features,threshold):

    image1_points = np.float32([key.pt for key in keypoints1])
    image2_points = np.float32([key.pt for key in keypoints2])
    
    image1_cordinate = np.float32([image1_points[key[0]] for key in matched_features]).reshape(-1, 1, 2)
    image2_cordinate = np.float32([image2_points[key[1]] for key in matched_features]).reshape(-1, 1, 2)
          
    (Homography_matrix, status) = cv2.findHomography(image1_cordinate, image2_cordinate, cv2.RANSAC, threshold)

    return Homography_matrix, status

def stitching_image(image1, image2, Homography_matrix ,showMatches=False, ratio=0.75):
    ideal_width = image2.shape[1] + image1.shape[1]
    ideal_height = max(image2.shape[0], image1.shape[0])
    output = cv2.warpPerspective(image2, Homography_matrix, (ideal_width, ideal_height))

    height, width, breadth = image1.shape
    for i in range(0, height):
        for j in range(0, width):
            if sum(list(image1[i, j])) and not sum(list(output[i, j])) :
                output[i, j] = image1[i, j]

    return output

def black_padding_removal(output):
    image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    l_side = np.argmax(np.any(image > 0, axis=0))
    r_side = image.shape[1] - np.argmax(np.any(image > 0, axis=0)[::-1])

    crop_image = output[:, l_side:r_side]

    return crop_image

def overlap_array(images):
    size = len(images)
    overlap_array = np.eye(size, dtype=int)
    for i in range(0,size):
        (keypoints1, features1) = extract_keypoint_feature_function(images[i])
        for j in range(i+1,size):
            (keypoints2, features2) = extract_keypoint_feature_function(images[j])
            matched_features = matching_featues(features1, features2)
            overlap_percentage = len(matched_features) / min(len(features1), len(features2))

            if overlap_percentage > 0.2:
                overlap_array[i, j] = 1
                overlap_array[j, i] = 1
            else:
                pass
    return overlap_array

if __name__ == "__main__":
    #task2
    args = parse_args()
    overlap_arr = stitch(args.input_path, 't2', N=4, savepath=f'{args.output_panaroma}')
    with open(f'{args.output_overlap}', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)

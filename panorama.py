import os
import sys
import cv2
import math
import numpy as np
import utils
from PIL import Image
from numpy import linalg

class Sticher:
    def stich(self, image1, image2):
        (kps1, feature1) = self.detector(image1)
        (kps2, feature2) = self.detector(image2)
        M = self.matchKeypoints(kps1, kps2, feature1, feature2)
        if M is None:
            return None
        (matches, H, status) = M

        ################## new stuff ############################

        inlierRatio = float(np.sum(status)) / float(len(status))
        closestImage = None
        if (closestImage == None or inlierRatio > closestImage['inliers']):
            closestImage = {}
            closestImage['h'] = H
            closestImage['inliers'] = inlierRatio

        H = closestImage['h']
        H = H / H[2, 2]
        H_inv = linalg.inv(H)

        if (closestImage['inliers'] > 0.1):  # and

            (min_x, min_y, max_x, max_y) = findDimensions(image2, H_inv)

            # Adjust max_x and max_y by base img size
            max_x = max(max_x, image1.shape[1])
            max_y = max(max_y, image1.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            mod_inv_h = move_h * H_inv
            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

        base_img_warp = cv2.warpPerspective(image1, move_h, (img_w, img_h))
        next_img_warp = cv2.warpPerspective(image2, mod_inv_h, (img_w, img_h))
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        # result = cv2.warpPerspective(image1, H, (3000, 3000))
        # result[0:image2.shape[0], 0:image2.shape[1]] = image2

        (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                        0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                    mask=np.bitwise_not(data_map),
                                    dtype=cv2.CV_8U)
        final_img = cv2.add(enlarged_base_img, next_img_warp,
                            dtype=cv2.CV_8U)

        return final_img

    def rank(self, image1, image2):
        (kps1, feature1) = self.detector(image1)
        (kps2, feature2) = self.detector(image2)
        M = self.matchKeypoints(kps1, kps2, feature1, feature2)
        if M is None:
            return None
        (matches, H, status) = M
        return len(matches)

    def detector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Difference of Gaussian
        sift = cv2.FeatureDetector_create("SIFT")
        # SIFT feature extractor
        kps = sift.detect(gray)
        result = cv2.DescriptorExtractor_create("SIFT")
        (kps, features) = result.compute(gray, kps)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kps1, kps2, features1, features2):
        ratio = 0.75
        reprojThresh = 4.0
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(features1, features2, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            pts1 = np.float32([kps1[i] for (_, i) in matches])
            pts2 = np.float32([kps2[i] for (i, _) in matches])
            # compute Homograpy of photo
            (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        # less then four no matches
        return None


def findDimensions(image, homography):

    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

        if (max_x == None or normal_pt[0, 0] > max_x):
            max_x = normal_pt[0, 0]

        if (max_y == None or normal_pt[1, 0] > max_y):
            max_y = normal_pt[1, 0]

        if (min_x == None or normal_pt[0, 0] < min_x):
            min_x = normal_pt[0, 0]

        if (min_y == None or normal_pt[1, 0] < min_y):
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)
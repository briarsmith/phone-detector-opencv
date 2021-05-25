import imutils
import cv2
from os import path
import numpy as np
import sys
from enum import Enum

class cntThresh(Enum):
    MIN_AREA = 150
    MAX_AREA = 850
    MIN_ARC_LEN = 50
    MAX_ARC_LEN = 140
    MIN_VERT = 3
    MAX_VERT = 6

class FindPhone:
    def __init__(self):
        if not len(sys.argv) > 1:
            print('Error: No path passed. Please pass a path as the first argument')
            return

        self.p = sys.argv[1]
        if not path.exists(self.p):
            print('Error: the path passed does not exist')
            return

        self.run()
        
    # convert a regular image to a thresholded one
    def thresh_img(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    def get_contours(self, thresh):
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)
        return imutils.grab_contours(cnts)

    # retreive geometric factors of a contour
    def get_geometry(self, c):
        area = cv2.contourArea(c)
        arcLen = cv2.arcLength(c,True)
        rect = cv2.minAreaRect(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        return area, arcLen, rect, peri, len(approx)

    # obtain the normalized centroid of a contour
    def get_norm_centroid(self, c):
        M = cv2.moments(c)
        cX, cY = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        return cX/490, cY/326

    # detect the phone centroid in the passed image
    def run(self):
        thresh = self.thresh_img(cv2.imread(self.p))
        cnts = self.get_contours(thresh)

        currMinVert = sys.maxsize
        phoneCnt = np.array([])
        # Check every contour to see if it meets the phone geometric thresholds
        for c in cnts:

            # Wish to avoid contours that will cause division by 0 to get the centroid
            if not cv2.moments(c)['m00']:
                continue

            area, arcLen, rect, peri, vert = self.get_geometry(c)
            phoneThreshMet = area < cntThresh.MAX_AREA.value and area > cntThresh.MIN_AREA.value 
            phoneThreshMet = phoneThreshMet and arcLen > cntThresh.MIN_ARC_LEN.value and arcLen < cntThresh.MAX_ARC_LEN.value 
            phoneThreshMet = phoneThreshMet and vert >= cntThresh.MIN_VERT.value and vert <= cntThresh.MAX_VERT.value
            
            if not phoneThreshMet:
                continue

            # choose the phone contour with the lowest amount of verticies as the final cont
            if vert < currMinVert:
                currMinVert = vert
                phoneCnt = c

        if cnts and phoneCnt.size: 
            normcX, normcY = self.get_norm_centroid(phoneCnt)
            print('{:.4f} {:.4f}'.format(normcX, normcY))
        else:
            print('{} {}'.format(-1,-1))

FindPhone()

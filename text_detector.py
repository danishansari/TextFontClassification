import os
import cv2
import math
import numpy as np
import argparse


'''
    @brief: Class to detect text bounding boxes from images
'''
class TextDetector(object):

    '''
        @brief: Function to calculate IOU
    '''
    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0]+box1[2], box2[0]+box2[2])
        y2 = min(box1[1]+box1[3], box2[1]+box2[3])
        eps = 1.0e-7
        area_inter = (x2-x1) * (y2-y1)
        area_union = (box1[2]*box1[3])+(box2[2]*box2[3])-area_inter
        return area_inter/(area_union+eps)

    '''
        @brief: Function to match IOU and get ground truth labels
    '''
    def get_labeled_boxes(self, detection, labeled_boxes):
        detections = []
        for i, det in enumerate(detection):
            lab, box1 = det
            for label, box2 in labeled_boxes: 
                if self.iou(box1, box2) > 0.6:
                    detections.append([label, box1])
        return detections

    '''
        @brief: Function to find eucleadean distance of two points
    '''
    def euc_dist(self, pt1, pt2):
        return math.sqrt(((pt1[0]-pt2[0])**2)+((pt1[1]-pt2[1])**2))

    def get_bdbox(self, points_list, pfw=0.0, pfh=0.0):
        if len(points_list) == 0:
            return []
        x1 = min([pt[0] for pt in  points_list])
        y1 = min([pt[1] for pt in  points_list])
        x2 = max([pt[0] for pt in  points_list])
        y2 = max([pt[1] for pt in  points_list])
        pw = int((x2-x1)*pfw)
        ph = int((y2-y1)*pfh)
        return [max(0, x1-int(pw/2)), max(0, y1-int(ph/2)), x2-x1+pw, y2-y1+ph]
    
    '''
        @brief: Function to combine small predictions into the biggest possible bdbox
    '''
    def combine_boxes(self, bdbox_list, img):
        global AVG_WIDTH_g
        global AVG_HEIGHT_g
        global TOTAL_SAMPLES_g
    
        detection_list = []
        y_boxes = sorted(bdbox_list, key=lambda x: x[1])
        y_cluster, ybox = [], []
        prev_added = 0
        # sort detection based on y-axis
        for i in range(1, len(y_boxes)):
            if len(ybox) == 0:
                ybox.append(y_boxes[i-1])
            if abs(ybox[-1][1] - y_boxes[i][1]) < (ybox[-1][3]):
                ybox.append(y_boxes[i])
            else:
                y_cluster.append(ybox)
                ybox = [y_boxes[i]]
        y_cluster.append(ybox)
        # sort boxes in a row on x axis and combine close boxes into one
        for boxes in y_cluster:
            x_boxes = sorted(boxes, key=lambda x: x[0])
            comb_points = []
            for i in range(1, len(x_boxes)):
                if len(comb_points) == 0:
                    comb_points.append([x_boxes[i-1][0], x_boxes[i-1][1]])
                    comb_points.append([x_boxes[i-1][0]+x_boxes[i-1][2], x_boxes[i-1][1]+x_boxes[i-1][3]])
                if abs(comb_points[-1][0]-x_boxes[i][0]) < 100:
                    comb_points.append([x_boxes[i][0], x_boxes[i][1]])
                    comb_points.append([x_boxes[i][0]+x_boxes[i][2], x_boxes[i][1]+x_boxes[i][3]])
                else:
                    bdbox = self.get_bdbox(comb_points, 0.1, 0.2)
                    if len(bdbox) > 0:
                        detection_list.append([-1, bdbox])
                    comb_points = []
                    comb_points.append([x_boxes[i][0], x_boxes[i][1]])
                    comb_points.append([x_boxes[i][0]+x_boxes[i][2], x_boxes[i][1]+x_boxes[i][3]])
            bdbox = self.get_bdbox(comb_points, 0.1, 0.2)
            if len(bdbox) > 0:
                detection_list.append([-1, bdbox])
        return detection_list
            
    '''
        @brief: Function to detect text bounding box from images
    '''
    def detect(self, image, label_boxes=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        can = cv2.Canny(gray, 30, 200)
        cimg, contours, hierarchy = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bdbox_list = []
        close_contour = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 65536:
                continue
            if len(close_contour) == 0:
                close_contour.append(list(cnt[0][0]))
            prev_added = 0
            for c in range(1, len(list(cnt))):
                if self.euc_dist(close_contour[-1], cnt[c-1][0]) < 25:
                    close_contour.append(list(cnt[c-1][0]))
                    prev_added = 1
                else:
                    if prev_added == 1:
                        close_contour.append(list(cnt[c][0]))
                        prev_added = 0
                    if len(close_contour) > 4:
                        bdbox = self.get_bdbox(close_contour)
                        bdbox_list.append(bdbox)
                    close_contour = [list(cnt[c][0])]
        detections = self.combine_boxes(bdbox_list, image)
        if label_boxes:
            detections = self.get_labeled_boxes(detections, label_boxes)
        return detections


##
# @author: Danish Ansari
# @date:   2021/01/24

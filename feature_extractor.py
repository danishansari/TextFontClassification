import os
import cv2
from tqdm import tqdm 
from skimage.feature import hog


'''
    @brief: Class to extract HOG Features for detected text
'''
class HOGFeatures(object):

    '''
        @brief: Initializer function, sets width, height and bins
    '''
    def __init__(self, width, height, bins=16):
        self.n_bins = bins
        self.img_w = width
        self.img_h = height

    '''
        @brief: Function to convert yolo annotations back to bdbox
    '''
    def convert_yolo2bdbox(self, box, W, H):
        w = int(box[2]*W)
        h = int(box[3]*H)
        x = int((box[0]*W) - (w/2))
        y = int((box[1]*H) - (h/2))
        return [x, y, w, h]

    '''
        @brief: Function to get yolo-annotation from txt files
    '''
    def get_bdboxes(self, filepath, W, H):
        bd_boxes = []
        with open(filepath) as fp:
            for lines in fp.readlines():
                val = lines.split()
                label = int(val[0])
                box = list(map(float, val[1:]))
                cbox = self.convert_yolo2bdbox(box, W, H)
                bd_boxes.append([label, cbox])
        return bd_boxes

    '''
        @brief: Function to extract hog features for an image
    '''
    def extract_hog(self, image, orient=9, pix_per_cell=8, cell_per_block=2):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_w, self.img_h))
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       feature_vector=True)
        return features

    '''
        @brief: Function to extract hog features and labels for all images
    '''
    def get_hog_features(self, path, detector=None, mode='TEST'):
        features, labels, boxes = [], [], []
        all_files = [path]
        if os.path.isdir(path):
            all_files = [os.path.join(path, x) for x in os.listdir(path) if '.jpg' in x or '.png' in x]
        print ('Extracting %s features...'%mode)
        for i in tqdm(range(len(all_files))):
            if '.jpg' in all_files[i] or '.png' in all_files[i]:
                filepath = all_files[i]
                img = cv2.imread(filepath)
                if img is None:
                    continue
                H, W = img.shape[:2]
                if mode != 'TEST':
                    if '.jpg' in filepath:
                        txt_file = filepath.replace('.jpg', '.txt')
                    elif '.png' in filepath:
                        txt_file = filepath.replace('.png', '.txt')
                    else:
                        print ('file %s format not supported!!', filepath)
                        continue
                    boxes = self.get_bdboxes(txt_file, W, H)
                elif detector is not None:
                    boxes = detector.detect(img)
                for val in boxes:
                    b = val[1]
                    img_roi = img[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
                    label = val[0]
                    feats = self.extract_hog(img_roi)
                    features.append(feats)
                    labels.append(label)
        print ('Feature extraction finshed.')
        return features, labels, boxes


##
# @author: Danish Ansari
# @date:   2021/01/24

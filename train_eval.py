import os
import sys
import cv2
import pickle
import numpy as np

from feature_extractor import HOGFeatures
from text_detector import TextDetector

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

labels_g = []
def load_labels():
    global labels_g
    labels_g = open('data/fonts.names').read().strip().split('\n')
    print (labels_g)

def train_svm():
    h = HOGFeatures(192, 32)
    d = TextDetector()
    features_trn, lables_trn, _ = h.get_hog_features('data/train', d, 'TRAIN')
    features_tst, lables_tst, _ = h.get_hog_features('data/test', d, 'EVAL')
    print ('Training started..')
    model = SVC(kernel='linear', probability=True)
    model.fit(features_trn, lables_trn)
    # Save the model
    pickle.dump(model, open('font_classification.sav', 'wb'))
    print(" Evaluating classifier on test data ...")
    predictions = model.predict(features_tst)
    print(classification_report(lables_tst, predictions))

def eval_svm():
    h = HOGFeatures(192, 32)
    d = TextDetector()
    features_val, lables_val, _ = h.get_hog_features('data/eval', d, 'EVAL')
    model = pickle.load(open('models/font_classification.sav', 'rb'))
    predictions = model.predict(features_val)
    print(classification_report(lables_val, predictions))

def test(filepath, show=False):
    global labels_g
    all_file = [filepath]
    if os.path.isdir(filepath):
        all_file = [os.path.join(filepath, x) for x in os.listdir(filepath) if '.jpg' in x or '.png' in x]
    h = HOGFeatures(192, 32)
    d = TextDetector()
    model = pickle.load(open('models/font_classification.sav', 'rb'))
    output = {'detectedFonts': []}
    for files in all_file:
        features, labels, boxes = h.get_hog_features(files, d)
        probabilities = model.predict_proba(features)
        for proba in probabilities:
            prediction = np.argmax(proba)
            print (proba, prediction)
            confidense = proba[prediction]
            prediction_str = labels_g[prediction]
            print ('Prediction:', prediction_str, confidense)
            det_dict = {}
            for b in boxes:
                det_dict = {"boundingBox": {"x": b[1][0], "y":b[1][1], "w": b[1][2], "h": b[1][3]}}
                det_dict["font"] = prediction_str
                det_dict["confidence"] =  confidense
                output["detectedFonts"].append(det_dict)
                if show:
                    img = cv2.imread(files)
                    cv2.rectangle(img, (b[1][0], b[1][1]), (b[1][0]+b[1][2], b[1][1]+b[1][3]), (255, 255, 0))
                    cv2.putText(img, '%s'%prediction_str, (b[1][0]-10, b[1][1]-10), 1, 1, (255, 255, 0))
    print ('output:', output)            
    if show:
        cv2.imshow('image', img)
        cv2.waitKey(0)

def main():
    if sys.argv[-1] == '-train':
        train_svm()
    elif sys.argv[-1] == '-eval':
        eval_svm()
    elif sys.argv[-1] == '-test' or sys.argv[-1] == '-show':
        load_labels()
        test_path = 'data/test'
        if len(sys.argv) > 2:
            test_path = sys.argv[1]
        show = False
        if sys.argv[-1] == '-show':
            show = True
        test(test_path, show)
    else:
        print ('error: bad option!!')
0
if __name__=='__main__':
    main()

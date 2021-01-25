import os
import sys
import cv2
import random

def convert(box, W, H):
    w = int(box[2]*W)
    h = int(box[3]*H)
    x = int((box[0]*W)-(w/2))
    y = int((box[1]*H)-(h/2))
    return [x, y, w, h]

def main():
    color_list = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(10)]
    for files in os.listdir(sys.argv[1]):
        txt_file = os.path.join(sys.argv[1], files)
        if '.txt' in txt_file:
            img_file = txt_file.replace('.txt', '.jpg')
            img = cv2.imread(img_file)
            print ('img file:', img_file, img.shape)
            boxes = []
            for line in open(txt_file).readlines():
                box = list(map(float, line.strip().split()))
                #box = list(map(int, line.strip().split()))
                cbox = convert(box[1:], img.shape[1], img.shape[1])
                boxes.append([int(box[0]), cbox])
            for box in boxes:
                lab = box[0]
                bbx = box[1]
                print (lab, bbx)
                cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0]+bbx[2], bbx[1]+bbx[3]), color_list[lab], 2)
            cv2.imshow('image', img)
            cv2.waitKey(0)

if __name__=='__main__':
    main()

import os
import sys
import cv2
import random
import numpy as np
import threading

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tqdm import tqdm 


all_fonts_g = []
font_names_g = []
font_label_g = {}


def load_all_fonts(path='fonts'):
    global all_fonts_g
    global font_names_g
    global font_label_g
    dirs = sorted(os.listdir(path))
    for i, d in enumerate(dirs):
        for files in os.listdir(os.path.join(path, d)):
            if '.ttf' in files.lower():
                all_fonts_g.append(os.path.join(path, d, files))
                font_names_g.append(d)
                font_label_g[d] = i
    with open('data.names', 'w') as fp:
        for d in dirs:
            fp.write('%s\n'%d)


def get_random_font():
    global all_fonts_g
    global font_names_g
    c = random.randint(0, len(all_fonts_g)-1)
    s = random.randint(20, 50)
    return ImageFont.truetype(all_fonts_g[c], s), font_names_g[c]


def gen_random_samples(img_w=512, img_h=512, start=0, n_samples=10, path='data'):
    global all_fonts_g
    global font_names_g
    global font_label_g

    text_string = 'Hello World!'
    init_pad = 10
    for n in tqdm(range(start, start+n_samples)): 
        img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        out = open('%s/sample_%06d.txt'%(path, n), 'w')
        bdbox = None
        x, y = random.randint(5, 15), random.randint(5, 15)
        while True:
            tmp = Image.new('RGB', (img_w, img_h), (0, 0, 0))
            dra = ImageDraw.Draw(tmp)
            font, name = get_random_font()
            text_w, text_h = font.getsize(text_string)
            pad = random.uniform(1.1, 1.3) # 10-30%
            box_w, box_h = int(text_w*pad), int(text_h*pad)
            if bdbox is None:
                x, y = random.randint(init_pad, init_pad+box_w-text_w), random.randint(init_pad, init_pad+box_h-text_h)
            else:
                x = random.randint(bdbox[2], bdbox[2]+box_w-text_w)
                y = random.randint(bdbox[3], bdbox[3]+box_h-text_h)
                if x+text_w+init_pad >= img_w and y+text_h+init_pad >= img_h:
                    break # no space in image left
                elif x+text_w+init_pad >= img_w and y+text_h+init_pad < img_h:
                    y += box_h
                    x = random.randint(init_pad, init_pad+box_w-text_w)
                elif x+text_w+init_pad < img_w and y+text_h+init_pad >= img_h:
                    x += box_w
                else: 
                    pass
            if x+text_w+init_pad >= img_w or y+text_h+init_pad >= img_h:
                break
            draw.text((x, y), text_string, (0, 0, 0), font=font)
            dra.text((x, y), text_string, (255, 255, 255), font=font)
            bdbox = tmp.getbbox()
            w = bdbox[2] - bdbox[0]
            h = bdbox[3] - bdbox[1]
            pad_w, pad_h = w*0.01, h*0.1
            x = (bdbox[0] - (pad_w/2) + (w/2))/img_w
            y = (bdbox[1] - (pad_h/2) + (h/2))/img_h
            w = (bdbox[2] - bdbox[0] + (4*pad_w))/img_w
            h = (bdbox[3] - bdbox[1] + (4*pad_h))/img_h
            out.write('%d %f %f %f %f\n'%(font_label_g[name], x, y, w, h))
        img.save('%s/sample_%06d.jpg' % (path, n))
        out.close()


def main():
    n_samples = 10
    data_path = 'data'
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    load_all_fonts()
    print ('Generating %d samples..'%n_samples, 'n_part:', n_samples/4)
    n_part = int(n_samples/4)
    thread_list = []
    for i in range(0, n_samples, n_part):
        thread = threading.Thread(target=gen_random_samples, args=(512, 512, i, n_part, data_path,))
        thread_list.append(thread)
        thread.start()
    for thread in thread_list:
        thread.join()

if __name__=='__main__':
    main()

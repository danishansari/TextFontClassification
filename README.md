Text Detection and font classification
--------------------------------------

Texts are detected using contours and then a svm is trained on HOG features
for classifying each text into multiple(10) fonts.

# dependencies(worked on):
1. python 3.7.5
2. numpy 1.19.5
3. PIL 6.1.0
4. skimage 0.18.1
5. sklearn 0.24.1

# run test:
1. python font_detector.py <path/to/image.jpg> -test
2. python font_detector.py <path/to/image.jpg> -show
3. python font_detector.py <path/to/image/folder> -test
4. python font_detector.py <path/to/image/folder> -show

# generate data
1. python gen_data.py 
2. python gen_data.py <num_samples_to_generate>

# run eval:
1. python font_detector.py -eval

# run train:
1. python font_detector.py -train

##
# @author: Danish Ansari
# @date:   2021/01/25

#-*- coding:utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+'/comp/')

import ocr
#from card_attribution import att_search


import numpy as np
from PIL import Image
from glob import glob
import shutil

image_files = glob(os.getcwd()+'/test_images/*.*')

result_dir = os.getcwd()+'/test_result'

if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)


def predict():
    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))

        result, image_framed = ocr.model(image)

        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)


        for key in result:

            write_path = os.getcwd()+'/test_result/result.txt'
            result_name = image_file.split('/')[-1]
            result_name = result_name.split('.')[0]

            print(result_name+':'+result[key][1])

            f1 = open(write_path,'a')
            f1.write(result_name+':'+result[key][1])
            f1.write('\n')



predict()
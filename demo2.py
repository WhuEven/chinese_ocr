#-*- coding:utf-8 -*-
import os
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
from cv2 import cv2

#image_file = glob('2.jpg')

if __name__ == '__main__':
    # result_dir = './test_result'
    # if os.path.exists(result_dir):
    #     shutil.rmtree(result_dir)
    # os.mkdir(result_dir)

    #for image_file in sorted(image_files):
    #src = cv2.imread('images_4.png')
    #crop = src[ 303:827,219:1530]
    #cv2.imwrite('crop.jpg',crop)
    #image = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_RGB2BGR))
    image = np.array(Image.open('2.jpg').convert('RGB'))
    #t = time.time()
    result, image_framed = ocr.model(image)
    #output_file = os.path.join(result_dir, image_file.split('/')[-1])
    #Image.fromarray(image_framed).save(output_file)
    #print("Mission complete, it took {:.3f}s".format(time.time() - t))
    print("\nRecognition Result:\n")
    for key in result:
        # print(result[key][1])
        with open('result.txt','a') as f:
                f.write(result[key][1]+'\n')

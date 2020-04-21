#-*- coding:utf-8 -*-
import os
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
image_files = glob('./data/autobinarization/*')#获取文件夹下所有文件路径


if __name__ == '__main__':
    result_dir = './data/autobinarization_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)#删除文件夹下所有文件
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))#读取图像并转为np矩阵
        t = time.time()
        result, image_framed = ocr.model(image)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
            with open('result.txt','a') as f:
                f.write(result[key][1]+'\n')



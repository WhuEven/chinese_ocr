#-*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os


#测试线性变换方法效果
def xxbh(img):
    out = 2.0 * img
    out[out >255] = 255
    out = np.around(out)
    out = out.astype(np.uint8)
    cv2.imshow('xxbh',out)

#测试直方图正规化方法效果
def zztzgh(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    Imin, Imax = cv2.minMaxLoc(img)[:2]
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * img + b
    out = out.astype(np.uint8)
    cv2.imshow('zftzgh',out)

def zztzgh2(img):
    out = np.zeros(img.shape, np.uint8)
    cv2.normalize(img, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('zftzgh',out)

#伽马变换
def gamma_t(img):
    # 图像归一化
    fi = img / 255.0
    # 伽马变换
    gamma = 0.4
    out = np.power(fi, gamma)
    cv2.imshow("out", out)

#全局直方图均衡化
def qjzftjhh(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    equa = cv2.equalizeHist(img)
    cv2.imshow('zftjhh',equa)

#限制对比度的全局直方图均衡化
def jhh_limit(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    cv2.imshow('jhh_limit',dst)

#拉普拉斯算子清晰度提升
def qxdts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)#ksize是算子的大小，必须为1、3、5、7。默认为1。
    
    dst = cv2.convertScaleAbs(imageVar)
    dst = np.add(dst,gray)
    cv2.imshow('qxdts',dst)

#固定阈值二值化
def binary_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    threshold = 80#设定阈值
    maxval = 255
    # ret，out = cv2.threshold(gray,threshold,maxval, cv2.THRESH_BINARY)#常规0/255
    # ret，out = cv2.threshold(gray,threshold,maxval, cv2.THRESH_BINARY_INV)#反向255/0
    # ret, out = cv2.threshold(gray,threshold, maxval, cv2.THRESH_TRUNC)#屏蔽亮值至maxval，暗值不变
    # ret，out = cv2.threshold(gray,threshold,maxval, cv2.THRESH_TOZERO)#屏蔽暗值0，亮值不变
    # ret，out = cv2.threshold(gray,threshold,maxval, cv2.THRESH_TOZERO_INV)#屏蔽亮值0,暗值不变
    # ret, out2 = cv2.threshold(gray,threshold, maxval, cv2.THRESH_OTSU)#大津算法

    # out = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,5)#高斯自适应阈值
    out = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)#中值自适应阈值


    cv2.imshow('out1',out)
    # cv2.imshow('dajin',out2)
    
#滤波+二值化
def lvbo_binary(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    dst = cv2.GaussianBlur(gray,(5,5),0)#高斯滤波
    # dst = cv2.medianBlur(gray,5)#中值滤波
    
    out = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,5)#自适应阈值

    cv2.imshow('gauss',dst)
    cv2.imshow('out',out)





if __name__ == "__main__":
    # img_path = './data/ori/images2_5.png'
    # img = cv2.imread(img_path)
    # cv2.imshow('ori',img)

    # xxbh(img)#线性变化
    # zztzgh2(img)#直方图正规化
    # gamma_t(img)#伽马变换
    # qjzftjhh(img)#全局直方图均衡化
    # jhh_limit(img)#限制对比度的全局直方图均衡化
    # qxdts(img)#拉普拉斯算子清晰度提升
    # binary_img(img)#二值化方法
    # lvbo_binary(img)#滤波+二值化

    #批处理
    image_files = glob('./data/ori/*')
    result_dir = './data/autobinarization'
    for image_file in sorted(image_files):
        image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        output = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,9)#自适应阈值
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        cv2.imwrite(output_file, output)


    # cv2.waitKey()
    # cv2.destroyAllWindows()
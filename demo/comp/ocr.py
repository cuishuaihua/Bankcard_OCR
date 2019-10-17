#-*- coding:utf-8 -*-
import os
import sys
import cv2
from math import *
import numpy as np
from PIL import Image

import random
sys.path.append(os.getcwd()[:-5])


from ctpn.text_detect import text_detect
from ctpn.lib.fast_rcnn.config import cfg_from_file
from densenet.model import predict as keras_densenet

from Re_VGG16 import predict
def sort_box(box):
    """ 
    对box进行排序
    """
    # print("====box=====",box)
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):

    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))


    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]

    return imgOut

def charRec(img, text_recs, adjust=False):
   """
   加载OCR模型，进行字符识别
   # text_recs：文本框位置（列表形式）
   """
   results = {}  #最终的接受字典，容器

   xDim, yDim = img.shape[1], img.shape[0]

   for index, rec in enumerate(text_recs):

       xlength = int((rec[6] - rec[0]) * 0.1)
       ylength = int((rec[7] - rec[1]) * 0.2)

       if adjust:
           pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
           pt4 = (rec[4], rec[5])
       else:
           #过滤下异常框
           pt1 = (max(1, rec[0]), max(1, rec[1]))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
           pt4 = (rec[4], rec[5])

        #将弧度转换为角度
       degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

       # partImg为裁减出文本框区域的的图片
       partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4) #处理图片，水平校准

       '''
       t = random.randint(0,651651651654151656151651651)
       path = os.getcwd()+'/comp/loc_results/'+str(t)+'.jpg'
       szy = Image.fromarray(partImg)

       szy.save(path)
       '''


       # 过滤异常图片
       if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:
           continue


       IMG = Image.fromarray(partImg)
       IMG.save(os.getcwd()+"/comp/text_proposal.jpg")

       path = os.getcwd()+'/comp/text_proposal.jpg'

       if predict(path) != 1:
           continue


       #  模式“L”为灰色图像,图片转为灰度图片
       image = Image.fromarray(partImg).convert('L')


       text = keras_densenet(image)  #文本识别模块


       if len(text)<16:
           continue



       if len(text)>0:
           results[index] = [rec]
           results[index].append(text)

 
   return results

def model(img, adjust=False):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('../ctpn/ctpn/text.yml')


    text_recs, img_framed, img = text_detect(img)  #检测文本框
    # text_recs：文本框位置（列表形式），
    # img_framed：在原图上标注了文本框的图片，
    # img:原图

    text_recs = sort_box(text_recs)  #排序框

    result = charRec(img, text_recs, adjust) #OCR模型，进行字符识别


    return result, img_framed


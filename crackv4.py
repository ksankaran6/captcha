from __future__ import print_function
import sys
import argparse
import os
import cv2 as cv
import cv2
import pytesser
from matplotlib import pyplot as plt
import numpy as np

def imgToText(filename,im_mask,filein,fileout,im_plot_list):

    im = cv2.imread (filein,cv2.IMREAD_GRAYSCALE)
    im_plot_list.append ((im,"input"))
    if im is None:
        error ( filein + " image is not found ")
        sys.exit(1)
  
    if im_mask is not None:
        im = cv2.bitwise_xor(im,im_mask)
        im_plot_list.append ((im,"masked"))
    else:
         th,im = cv2.threshold(im,125,255,cv2.THRESH_BINARY_INV)
         im_plot_list.append ((im,"invert"))
    
    height, width  = im.shape[:2]
    im= cv2.resize(im,(width*8,height*8), interpolation = cv2.INTER_LANCZOS4)
    im_plot_list.append ((im,"resize-up"))


    im = cv2.GaussianBlur(im,(51,51),0)
    im_plot_list.append ((im,"smoothed")) 

    kernel = np.ones((7,7),np.uint8)
    im = cv2.dilate(im,kernel,iterations = 1)
    im = cv2.erode(im,kernel,iterations = 3)
    im = cv2.dilate(im,kernel,iterations = 1)
    im_plot_list.append ((im,"dilation")) 


    im = cv2.resize(im,(width,height), interpolation = cv2.INTER_LANCZOS4)
    im_plot_list.append ((im,"resize-dn"))

    th,im = cv2.threshold(im,40,255,cv2.THRESH_BINARY_INV)
    im_plot_list.append ((im,"threshold-final"))


    cv2.imwrite(fileout, im)
    return ocr(im) 	


def smoothImage(im, nbiter=0, filter=(3,3)):
    for i in range(nbiter):
        im = cv2.GaussianBlur(im,(51,51),0)
    return im

def plotimg (im_plot_list):
    items  = len (im_plot_list)
    i = 431 
    for im,desc in im_plot_list:
        #plt.subplot(i),plt.imshow(im),plt.get_cmap('gray'), plt.title(desc),vmin = 0, vmax = 255
        plt.subplot(i)
        plt.title(desc)
        plt.imshow(im, cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
        i = i +1
    plt.xticks([]), plt.yticks([])
    plt.show()

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)
def error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)

def ocrfile(filein):
    im = cv2.imread (filein,cv2.IMREAD_GRAYSCALE)
    return ocr(im)

def ocr(im):
    if im == None or im.size == 0: 
       raise ValueError("Could not load image or image is empty")
    res = pytesser.image_to_string(im)
    res = res[:-2] #Remove the two \n\n always put at the end of the result
    return res

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some captchas.')
    parser.add_argument('-d', dest='base_dir', type=str, help='input image base dir',default='captcha', required=False)
    parser.add_argument('-i', dest='input_dir', type=str, help='input image input dir (relative to base)',default='input', required=False)
    parser.add_argument('-o', dest='output_dir', type=str, help='output image  dir (relative to base)',default='output', required=False)
    parser.add_argument('-s', dest='mask_dir', type=str, help='image mask dir (relative to base)',default='mask', required=False)
    parser.add_argument('-m', dest='input_mask', type=str, help='input image mask',default='', required=False)
    parser.add_argument('-t', dest='test_flag', type=bool, help='process just one image for testing',default=False, required=False)
    parser.add_argument('-p', dest='no_processing', type=bool, help='Just do OCR no processing',default=False, required=False)


    args = parser.parse_args()

    args.input_dir = os.path.join(args.base_dir,args.input_dir)
    args.output_dir = os.path.join(args.base_dir,args.output_dir)
    args.mask_dir = os.path.join(args.base_dir,args.mask_dir)


    if (False == os.path.isdir(args.base_dir)):
        raise ValueError("Directory does not exist", args.base_dir) 
    if (False == os.path.isdir(args.input_dir)):
        raise ValueError("Directory does not exist", args.input_dir) 
    if (False == os.path.isdir(args.output_dir)):
        raise ValueError("Directory does not exist", args.output_dir) 

    im_plot_list = []
    im_mask = None

    if (args.input_mask != ''):
        secmask =  os.path.join(args.mask_dir,args.input_mask)
        im_mask = cv2.imread (secmask,cv2.IMREAD_GRAYSCALE)
        if im_mask == None or im_mask.size == 0: 
           print ("Image loaded is empty: " +secmask)
           sys.exit(1) 
        im_plot_list.append ((im_mask,"mask"))	
	
    for dirname, dirnames, filenames in os.walk(args.input_dir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            filein  =os.path.join(args.input_dir,filename)
            fileout = os.path.join(args.output_dir,name+'_o.png')
            txt = imgToText (filename,im_mask,filein,fileout,im_plot_list)
            #print (filename,filein,fileout,txt)
            print (filename,",",ocrfile(filein),",",txt)
            if (args.test_flag == True): 
                plotimg(im_plot_list) 
                break
            del im_plot_list[:]

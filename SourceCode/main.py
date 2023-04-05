import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import grayScale


def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    if (mor_op[0:2] == "GS"): #dùng thuật toán grayscale thì inputImage là ảnh độ xám
        img = img_gray
    else:
        (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('binary image', img)
        cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    img_out = None

    '''
    TODO: implement morphological operators
    '''
    # Toán tử dilation
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual

    # Toán tử erosion
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual

    # Toán tử openning
    elif mor_op == 'opening':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_opening_manual = binary.opening(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual

    # Toán tử closing
    elif mor_op == 'closing':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_closing_manual = binary.closing(img, kernel)
        cv2.imshow('manual close image', img_closing_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual

    # Toán tử Hit-or-Miss
    elif mor_op == 'hitOrMiss':
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        img_hitOrMiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('OpenCV hitOrMiss image', img_hitOrMiss)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_hitOrMiss_manual = binary.hitOrMiss(img, kernel)
        cv2.imshow('manual hitOrMiss image', img_hitOrMiss_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_hitOrMiss_manual

    # Toán tử làm mảnh
    elif mor_op == 'thinning':
        img_thinning = cv2.ximgproc.thinning(img, cv2.ximgproc.THINNING_GUOHALL)
        cv2.imshow('OpenCV thinning image', img_thinning)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_thinning_manual = binary.thinning(img)
        cv2.imshow('manual close image', img_thinning_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual

    # Toán tử boundary extraction
    elif mor_op == 'bounextra':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img_erosion = cv2.erode(img, kernel)
        img_bounextra = img_closing - img_erosion
        cv2.imshow('OpenCV boundary extraction image', img_bounextra)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_bounextra_manual = binary.boundary_extraction(img, kernel)
        cv2.imshow('manual boundary extraction image', img_bounextra_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_bounextra_manual

################################################################################################
    # Toán tử dilation độ xám
    elif mor_op == 'GSdilate':
        img_gs_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV grayscale dilation image', img_gs_dilation)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img_gs_dilation_manual = grayScale.dilate(img, kernel)
        cv2.imshow('manual grayscale dilation image', img_gs_dilation_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)
        
        img_out = img_gs_dilation_manual

    # Toán tử erosion độ xám
    elif mor_op == 'GSerode':
        img_gs_dilation = cv2.erode(img, kernel)
        cv2.imshow('OpenCV grayscale erosion image', img_gs_dilation)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img_gs_erosion_manual = grayScale.erode(img, kernel)
        cv2.imshow('manual grayscale erosion image', img_gs_erosion_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_gs_erosion_manual

    # Toán tử opening độ xám
    elif mor_op == 'GSopen':
        img_gs_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV grayscale opening image', img_gs_opening)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img_gs_opening_manual = grayScale.open(img, kernel)
        cv2.imshow('manual grayscale opening image', img_gs_opening_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_gs_opening_manual
    # Toán tử closing độ xám
    elif mor_op == 'GSclose':
        img_gs_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV grayscale closing image', img_gs_closing)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img_gs_closing_manual = grayScale.close(img, kernel)
        cv2.imshow('manual grayscale closing image', img_gs_closing_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_gs_closing_manual

    # Toán tử morphological gradient
    elif mor_op == 'GSgradient':
        img_gs_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)
        cv2.imshow('OpenCV grayscale gradient image', img_gs_gradient)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img_gs_gradient_manual = grayScale.gradient(img, kernel)
        cv2.imshow('manual grayscale gradient image', img_gs_gradient_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_gs_gradient_manual

    # Toán tử top-hat
    elif mor_op == 'GStophat':
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        img_gs_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV grayscale top hat image', img_gs_tophat)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img_gs_tophat_manual = grayScale.tophat(img, kernel)
        cv2.imshow('manual grayscale top hat image', img_gs_tophat_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_gs_tophat_manual

    # Toán tử black-hat
    elif mor_op == 'GSblackhat':
        img_gs_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV grayscale black hat image', img_gs_blackhat)#show ảnh do CV2 thực hiện
        cv2.waitKey(wait_key_time)

        img_gs_blackhat_manual = grayScale.blackhat(img, kernel)
        cv2.imshow('manual grayscale black hat image', img_gs_blackhat_manual)#show ảnh mannual
        cv2.waitKey(wait_key_time)

        img_out = img_gs_blackhat_manual

    # Toán tử textural segmentation
    elif mor_op == 'GStese':
        
        img_gs_tese_manual = grayScale.textual_segmentation(img, kernel)
        cv2.imshow('manual textural segmentation image', img_gs_tese_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_tese_manual
    if img_out is not None:
        cv2.imwrite(out_file, img_out * 255)


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'python main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey()

### Command Line: 
'''  
python main.py -i D://input//lenna.jpg -o D://output//dilate.jpg -p dilate -t 0
python main.py -i D://input//lenna.jpg -o D://output//erode.jpg -p erode -t 0
python main.py -i D://input//lenna.jpg -o D://output//opening.jpg -p opening -t 0
python main.py -i D://input//lenna.jpg -o D://output//closing.jpg -p closing -t 0
python main.py -i D://input//lenna.jpg -o D://output//hitOrMiss.jpg -p hitOrMiss -t 0
python main.py -i D://input//lenna.jpg -o D://output//thinning.jpg -p thinning -t 0
python main.py -i D://input//lenna.jpg -o D://output//bounextra.jpg -p bounextra -t 0

python main.py -i D://input//lenna.jpg -o D://output//GSdilate.jpg -p GSdilate -t 0
python main.py -i D://input//lenna.jpg -o D://output//GSerode.jpg -p GSerode -t 0
python main.py -i D://input//lenna.jpg -o D://output//GSopen.jpg -p GSopen -t 0
python main.py -i D://input//lenna.jpg -o D://output//GSclose.jpg -p GSclose -t 0
python main.py -i D://input//lenna.jpg -o D://output//GSgradient.jpg -p GSgradient -t 0
python main.py -i D://input//lenna.jpg -o D://output//GStophat.jpg -p GStophat -t 0
python main.py -i D://input//lenna.jpg -o D://output//GSblackhat.jpg -p GSblackhat -t 0
python main.py -i D://input//lenna.jpg -o D://output//GStese.jpg -p GStese -t 0
'''
if __name__ == "__main__":
    main(sys.argv[1:])
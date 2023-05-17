import cv2
import numpy as np


def dilate(img, kernel):
    row_padding = kernel.shape[0] // 2
    col_padding = kernel.shape[1] // 2 #tính kích thước để lưu ảnh
    dilate = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1), np.uint8) #tạo ma trận trống để lưu ảnh
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1)) #tạo ma trận x row=img, col=img-1
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            maxValue = 0 
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    if img[i + m, j + n] + kernel[m, n] > maxValue:
                        maxValue = min(img[i + m, j + n] + kernel[m, n],255) #nếu lớn hơn max value thì cập nhật lại max[ ,255]
            dilate[i + row_padding, j + col_padding] = maxValue
    return dilate[: dilate.shape[0] - 2*row_padding,: dilate.shape[1] - 2*col_padding]


def erode(img, kernel):
    row_padding = kernel.shape[0] // 2
    col_padding = kernel.shape[1] // 2 #tính kích thước để lưu ảnh
    erode = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1), np.uint8) #tạo ma trận trống để lưu ảnh
    img_shape = img.shape #lưu kích thước ảnh

    x_append = np.ones((img.shape[0], kernel.shape[1] - 1))*255 #tạo ma trận x row=img, col=img-1
    img = np.append(img, x_append, axis=1)

    y_append = np.ones((kernel.shape[0] - 1, img.shape[1]))*255
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            minValue = img[i, j] + kernel[0, 0]
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    if img[i + m, j + n] + kernel[m, n] < minValue:
                        minValue = max(img[i + m, j + n] - kernel[m, n],0) #nếu lớn hơn min value thì cập nhật lại min[0, ]
            erode[i + row_padding, j + col_padding] = minValue
    return erode[: erode.shape[0] - 2*row_padding,: erode.shape[1] - 2*col_padding]


def open(img, kernel):
    # thực hiện erosion rồi dilation
    erode_img = erode(img, kernel)
    open_img = dilate(erode_img, kernel)
    return open_img


def close(img, kernel):
    # thực hiện dilation rồi erosion
    dilate_img = dilate(img, kernel)
    close_img = erode(dilate_img, kernel)
    return close_img

def gradient(img, kernel):
    # thực hiện dilation và erosion
    dilate_img = dilate(img, kernel)
    erode_img = erode(img, kernel)
    # thực hiện trừ 2 ảnh cho ra ảnh kết quả
    gradient_img = dilate_img - erode_img
    return gradient_img

def tophat(img, kernel):
    #img_shape = img.shape
    # thực hiện mở độ xám
    opening_img = open(img, kernel)
    # lấy ảnh gốc trừ ảnh opening
    tophat_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > opening_img[i,j]: #nếu giá trị pixel ảnh lớn hơn opening thì 
                tophat_img[i,j] = img[i,j] - opening_img[i,j] #gán giá trị khác biệt cho tophat_img
    return tophat_img

def blackhat(img, kernel):
    img_shape = img.shape
    # thực hiện đóng độ xám
    closing_img = close(img, kernel)
    # lấy ảnh gốc trừ ảnh closing
    blackhat_img = np.zeros((img.shape[0], img.shape[1]), np.uint8) #tạo ma trận trống để lưu ảnh sau khi thực hiện
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i,j] < closing_img[i,j]: #nếu giá trị pixel ảnh nhỏ hơn closing thì 
                blackhat_img[i,j] = closing_img[i,j] - img[i,j] #gán giá trị khác biệt cho blackhat_img
    return blackhat_img

def textual_segmentation(img, kernel1, kernel2 = np.ones((3,3))):
    # thực hiện đóng độ xám với kernel 1
    closing_img = close(img, kernel1)
    # thực hiện mở độ xám với kernel 2
    textual_segmentation = open(closing_img, kernel2)

    return textual_segmentation

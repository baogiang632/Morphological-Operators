import cv2
import numpy as np

# Toán tử erosion
def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2) #tìm tâm của kernel
    kernel_ones_count = kernel.sum() 
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1)) #tạo ma trận trống để chưa ảnh sau khi co
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1)) #tạo ma trận x row=img, col=img-1
    img = np.append(img, x_append, axis=1) #gán x vào img theo chiều ngang

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0] #tính kích thước 
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum(): 
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1 #nếu tổng giá trị 1 trong kernel = kernel hiện tại thì gán giá trị 1 cho ero_img

    return eroded_img[:img_shape[0], :img_shape[1]]


'''
TODO: implement morphological operators
'''
# Phép lấy nghịch đảo 1 ảnh
def complement(img):
    return abs(np.array(img, copy=True) - 1)

# Phép giao 2 ảnh
def andding(img1, img2):
    result = np.zeros((img1.shape[0],img1.shape[1]))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j] == img2[i,j]:
                result[i,j] = img1[i,j]
    return result

# Phép trừ 2 ảnh
def subtracting(img1, img2):
    result = np.array(img1, copy=True)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j] == img2[i,j]:
                result[i,j] = 0
    return result

# Toán tử dilation
def dilate(img, kernel):
    dilate = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1)) #tạo ma trận trống để chưa ảnh sau khi giãn
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))#tạo ma trận x row=img, col=img-1
    img = np.append(img, x_append, axis=1)#gán x vào img theo chiều ngang

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i, j] == 1.0: #nếu giá trị pixel =1 thì 
                for m in range(kernel.shape[0]):
                    for n in range(kernel.shape[1]):
                        if kernel[m, n] == 1:
                            dilate[i + m, j + n] = 1 #nếu giá trị pixel=1 thì gán 1 cho dilate

    return dilate[:img_shape[0], :img_shape[1]]

# Toán tử opening
def opening(img, kernel):
    # thực hiện erosion rồi dilation
    erode_img = erode(img, kernel)
    open_img = dilate(erode_img, kernel)
    return open_img

# Toán tử closing
def closing(img, kernel):
    # thực hiện dilation rồi erosion
    dilate_img = dilate(img, kernel)
    close_img = erode(dilate_img, kernel)
    return close_img

# Toán tử Hit-Or-Miss
def hitOrMiss(img, kernel):
    # thực hiện tạo 2 mặt nạ, mặt nạ hit và mặt nạ miss
    kernel_hit = np.zeros((kernel.shape[0], kernel.shape[1]))
    kernel_miss = np.zeros((kernel.shape[0], kernel.shape[1]))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if kernel[i, j] == 1:
                kernel_hit[i, j] = 1 # =1 thì gán hit, =-1 thì gán miss
            if kernel[i, j] == -1:
                kernel_miss[i, j] = 1
    # erode ảnh gốc bởi mặt nạ hit
    dilate_kernel_hit = erode(img, kernel_hit)
    complement_img = complement(img)
    # erode ảnh nghịch đảo ảnh gốc bởi mặt nạ miss
    dilate_kernel_miss = erode(complement_img, kernel_miss)
    # lấy giao 2 ảnh trên thu được kết quả hit or miss
    hitOrMiss_img = andding(dilate_kernel_miss, dilate_kernel_hit)
    return hitOrMiss_img

# Toán tử thinning
def thinning(img):
    # tạo 8 ma trận
    kernels = [np.array([[-1, -1, -1],
                         [0, 1, 0],
                         [1, 1, 1]]),
               np.array([[0, -1, -1],
                         [1, 1, -1],
                         [1, 1, 0]]),
               np.array([[1, 0, -1],
                         [1, 1, -1],
                         [1, 0, -1]]),
               np.array([[1, 1, 0],
                         [1, 1, -1],
                         [0, -1, -1]]),
               np.array([[1, 1, 1],
                         [0, 1, 0],
                         [-1, -1, -1]]),
               np.array([[0, 1, 1],
                         [-1, 1, 1],
                         [-1, -1, 0]]),
               np.array([[-1, 0, 1],
                         [-1, 1, 1],
                         [-1, 0, 1]]),
               np.array([[-1, -1, 0],
                         [-1, 1, 1],
                         [0, 1, 1]])]
    # tạo 2 bản sao ảnh
    prev_img = np.array(img, copy=True)
    prev_loop_img = np.array(img, copy=True)
    while_loop = 0
    # Lặp đến khi 2 ảnh trước và sau không còn sự khác biệt
    while True:
        while_loop += 1
        print("loop: ", while_loop)
        for i in range(8):
            current_img = andding(prev_img, complement(hitOrMiss(prev_img, kernels[i])))
            prev_img = np.array(current_img, copy=True)
        if (current_img == prev_loop_img).all():
            break
        prev_loop_img = current_img

    return current_img

# Toán tử boundary extraction
def boundary_extraction(img, kernel):
    erode_img = erode(img, kernel) #thực hiện ero
    boundary_extraction_img = subtracting(img, erode_img)*255 #trừ ảnh gốc với ero rồi * với 255 để giá trị màu trong khoảng [0,255]
    return boundary_extraction_img
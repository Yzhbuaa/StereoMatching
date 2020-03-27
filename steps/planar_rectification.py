from steps.file_parser import parse_data
from matplotlib import pyplot as plt
from steps.eight_points import eight_points_algorithm
import numpy as np
from scipy import linalg
import math

import cv2

# this is just to unconfuse pycharm
try:
    from cv2 import cv2
except ImportError:
    pass


def planar_rectification(img1, img2, epipole1, epipole2, pts1_h, pts2_h, matrix_f):
    # FIRST STEP
    # find the center of each image
    center = []
    if img1.shape == img2.shape:
        print('img1.shape == img2.shape')
        center = np.array([img1.shape[1] / 2, img2.shape[0] / 2])
        center = np.int_(center)
    else:
        print('error: image1 and image2 have different shapes')

    # define T' the translational matrix
    matrix_t_prime = np.array([[1, 0, -center[0]],
                               [0, 1, -center[1]],
                               [0, 0, 1]])

    # define R' the rotational matrix

    epipole2_t_trans = matrix_t_prime @ epipole2

    if epipole2_t_trans[0] >= 0:
        a = 1
    else:
        a = -1

    r = np.sqrt(epipole2_t_trans[0]**2 + epipole2_t_trans[1]**2)

    # Attention! not right handed coordinate system.
    theta = np.arctan(epipole2_t_trans[1]/epipole2_t_trans[0])

    matrix_r_prime = np.array([[np.cos(theta),  np.sin(theta), 0],
                               [-np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])

    # rotation matrix used in 03-epipolar-geometry.pdf from CMU's CS231A class.
    matrix_r_prime = np.array([[a * epipole2_t_trans[0]/r, a*epipole2_t_trans[1]/r, 0],
                               [-a * epipole2_t_trans[1]/r,  a*epipole2_t_trans[0]/r, 0],
                               [0,0,1]])

    # define G' so that H' == G' @ R' @ T'
    epipole2_rt_trans = matrix_r_prime @ epipole2_t_trans
    scalar_f = epipole2_rt_trans[0]

    matrix_g_prime = np.array([[1 , 0, 0],
                               [0, 1, 0],
                               [-1 / scalar_f, 0, 1]])

    # define H'
    matrix_h_prime = matrix_g_prime @ matrix_r_prime @ matrix_t_prime

    # SECOND STEP
    matrix_e_prime = np.array([[0, -epipole2[2], epipole2[1]],
                               [epipole2[2], 0, -epipole2[0]],
                               [-epipole2[1], epipole2[0], 0]])

    # proved in 03-epipolar-geometry.pdf
    # mm = matrix_e_prime - matrix_e_prime@matrix_e_prime@matrix_e_prime/(matrix_e_prime@matrix_e_prime@matrix_e_prime)[0,1] * (-1)
    matrix_m = matrix_e_prime @ matrix_f
    scalar_matrix = np.reshape(epipole2, (3, 1)) @ np.array([[1, 1 ,1]]) # todo: why?
    matrix_m = matrix_m + scalar_matrix

    # pts1_hat and pts2_hat
    pts1_hat = []
    pts2_hat = []
    for pt1, pt2 in zip(pts1_h, pts2_h):
        pt1_hat = matrix_h_prime @ matrix_m @ pt1
        pt1_hat = pt1_hat / pt1_hat[-1]
        pt2_hat = matrix_h_prime @ pt2
        pt2_hat = pt2_hat / pt2_hat[-1]
        pts1_hat.append(pt1_hat)
        pts2_hat.append(pt2_hat)

    # solve the minimization problem on page 237 of MVG HZ
    matrix_a = np.array(pts1_hat)
    array_b = np.array(pts2_hat)[:, 0]

    print(np.linalg.matrix_rank(matrix_a))

    # algorithm A5.1 on page 458 of MVG HZ
    u, s, vh = np.linalg.svd(matrix_a)
    array_b_prime = np.transpose(u) @ array_b

    array_y = []
    for bi_prime, di in zip(array_b_prime, s):
        yi = bi_prime / di
        array_y.append(yi)

    array_y = np.array(array_y)

    abc = np.transpose(vh) @ array_y

    # matrix H
    matrix_ha = np.array([[abc[0], abc[1], abc[2]],
                          [0, 1, 0],
                          [0, 0, 1]])
    matrix_h = matrix_ha @ matrix_h_prime @ matrix_m

    print('H = ')
    print(matrix_h)
    print(' ')
    print("H' = ")
    print(matrix_h_prime)
    print(' ')

    return matrix_h, matrix_h_prime

    print('debug')


def create_blank_image(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((int(height), int(width), 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def generate_blank_image_size(img_to_be_rectified, matrix_h):
    # ensure the rectified images side length
    img_width = img_to_be_rectified.shape[1]
    img_height = img_to_be_rectified.shape[0]

    a = np.array([0, 0, 1])
    b = np.array([img_width-1, 0, 1])
    c = np.array([img_width - 1, img_height - 1, 1])
    d = np.array([0, img_height - 1, 1])
    abcd = np.array([a, b, c, d])

    abcd_trans = []
    for i in abcd:
        i_trans = matrix_h @ i
        i_trans = i_trans / i_trans[-1]
        abcd_trans.append(i_trans)

    max_y =max(abcd_trans[0][1], abcd_trans[1][1], abcd_trans[2][1], abcd_trans[3][1])
    max_x =max(abcd_trans[0][0], abcd_trans[1][0], abcd_trans[2][0], abcd_trans[3][0])
    min_y = min(abcd_trans[0][1], abcd_trans[1][1], abcd_trans[2][1], abcd_trans[3][1])
    min_x = min(abcd_trans[0][0], abcd_trans[1][0], abcd_trans[2][0], abcd_trans[3][0])


    return max_x, max_y, min_x, min_y


def generate_rectified_image(img_to_be_rectified, matrix_h, blank_img, min_x, min_y):
    # # ensure the rectified images side length
    img_width = img_to_be_rectified.shape[1]
    img_height = img_to_be_rectified.shape[0]
    #
    # a = np.array([0, 0, 1])
    # b = np.array([img_width-1, 0, 1])
    # c = np.array([img_width - 1, img_height - 1, 1])
    # d = np.array([0, img_height - 1, 1])
    # abcd = np.array([a, b, c, d])
    #
    # abcd_trans = []
    # for i in abcd:
    #     i_trans = matrix_h @ i
    #     i_trans = i_trans / i_trans[-1]
    #     abcd_trans.append(i_trans)
    #
    #
    # max_y =max(abcd_trans[0][1], abcd_trans[1][1], abcd_trans[2][1], abcd_trans[3][1])
    # max_x =max(abcd_trans[0][0], abcd_trans[1][0], abcd_trans[2][0], abcd_trans[3][0])
    # min_y = min(abcd_trans[0][1], abcd_trans[1][1], abcd_trans[2][1], abcd_trans[3][1])
    # min_x = min(abcd_trans[0][0], abcd_trans[1][0], abcd_trans[2][0], abcd_trans[3][0])
    #
    # rectified_img_height = max_y -min_y+1
    # rectified_img_width = max_x - min_x +1
    #
    # rectified_img = create_blank_image(rectified_img_width,rectified_img_height,(255,255,255))

    rectified_img = blank_img
    for i in range(0, img_width):
        for j in range(0, img_height):
            px_trans = []
            px_trans = matrix_h @ np.array([i, j, 1])
            px_trans = px_trans / px_trans[-1]
            px_trans[0] = px_trans[0] - min_x
            px_trans[1] = px_trans[1] - min_y
            print(i, j)

            rectified_img[int(px_trans[1]),int(px_trans[0])] = img_to_be_rectified[j, i]

    return rectified_img


def main():
    # read images
    img1 = cv2.imread('../data/img1_with_8epipolar_lines.jpg')  # left image
    b1, g1, r1 = cv2.split(img1)
    img1 = cv2.merge([r1, g1, b1])
    img2 = cv2.imread('../data/img2_with_8epipolar_lines.jpg')  # right image
    b2, g2, r2 = cv2.split(img2)
    img2 = cv2.merge([r2, g2, b2])

    sixteen_points = parse_data(base_path='../data/', file_name='eight_points_qu')

    # slices image points in left and right images, non-homogeneous(nh)
    pts1_nh = sixteen_points[0:16:2]
    pts2_nh = sixteen_points[1:16:2]
    pts1_nh = np.int_(pts1_nh)
    pts2_nh = np.int_(pts2_nh)

    # transform non-homogeneous coordinates into their homogeneous form
    pts1_h = np.column_stack((pts1_nh, np.ones(pts1_nh.shape[0])))
    pts2_h = np.column_stack((pts2_nh, np.ones(pts2_nh.shape[0])))

    pts1_h = np.int_(pts1_h)
    pts2_h = np.int_(pts2_h)

    # calculate fundamental matrix F
    matrix_f = eight_points_algorithm(sixteen_points)
    print('Fundamental matrix:')
    print(matrix_f)

    # calculates epipoles
    epipole1 = linalg.null_space(matrix_f)  # left null space of matrix_f
    epipole2 = linalg.null_space(matrix_f.T)  # right null space of matrix_f
    epipole1 = np.reshape(epipole1, (-1))
    epipole2 = np.reshape(epipole2, (-1))
    epipole1 = epipole1 / epipole1[-1]
    epipole2 = epipole2 / epipole2[-1]
    print('Epipole1:')
    print(epipole1)
    print('Epipole2:')
    print(epipole2)

    matrix_h, matrix_h_prime = planar_rectification(img1, img2, epipole1, epipole2, pts1_h, pts2_h, matrix_f)

    max_x1, max_y1,min_x1, min_y1 = generate_blank_image_size(img1, matrix_h)
    max_x2, max_y2,min_x2, min_y2 = generate_blank_image_size(img2, matrix_h_prime)

    max_x = max(max_x1, max_x2)
    max_y = max(max_y1,max_y2)
    min_x = min(min_x1, min_x2)
    min_y = min(min_y1, min_y2)

    blank_img_height = max_x-min_x + 2
    blank_img_width = max_y - min_y + 2
    blank_img1 = create_blank_image(blank_img_width,blank_img_height,(255,255,255))
    blank_img2 = create_blank_image(blank_img_width,blank_img_height,(255,255,255))

    img1_rectified = generate_rectified_image(img1, matrix_h, blank_img1, min_x, min_y)
    plt.subplot(121)
    plt.imshow(img1_rectified)
    plt.show()

    img2_rectified = generate_rectified_image(img2, matrix_h_prime, blank_img2, min_x, min_y)
    plt.subplot(121)
    plt.imshow(img1_rectified)
    plt.subplot(122)
    plt.imshow(img2_rectified)
    plt.show()

    print('break point')


if __name__ == '__main__':
    main()

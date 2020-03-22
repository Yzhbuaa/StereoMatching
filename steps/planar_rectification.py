#  implementation of Marc Pollefeys's polar rectification algorithm)

from steps.file_parser import parse_data
from matplotlib import pyplot as plt
from steps.eight_points import eight_points_algorithm
import numpy as np
from scipy import linalg
from utils.coordinate_transform import homogeneous_coordinate_calc

import cv2

# this is just to unconfuse pycharm
try:
    from cv2 import cv2
except ImportError:
    pass


# computes epopolar lines using fundamental matrix
# Parameters:
#   pts - homogeneous form points in one image(left or right).
#   which_image - Index of the image (1 or 2) that contains the points.
#   matrix_f - fundamental matrix x'Fx = 0.
# Returns:
#   corresponding epipolar lines in another image.
def compute_correspond_epilines(pts, which_image, matrix_f):
    epilines = []
    if which_image == 1:
        for pt in pts:
            epilines.append(matrix_f @ pt)
    elif which_image == 2:
        for pt in pts:
            epilines.append(matrix_f.T @ pt)
    else:
        return
    # todo: add judgements of false input
    return np.array(epilines)


# draw epipolar lines using opencv
def draw_lines_cv2(img1, img2, lines1, lines2, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        pts1 - points on img1
        lines - corresponding epilines '''
    height, width, color_channel_number = img1.shape
    for line1, line2, pt1, pt2 in zip(lines1, lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line1[2] / line1[1]])
        x1, y1 = map(int, [width, -(line1[2] + line1[0] * width) / line1[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv2.circle(img1, tuple([pt1[0],pt1[1]]), 5, color, -1)

        x2, y2 = map(int, [0, -line2[2] / line2[1]])
        x3, y3 = map(int, [width, -(line2[2] + line2[0] * width) / line2[1]])
        img2 = cv2.line(img2, (x2, y2), (x3, y3), color, 2)
        img2 = cv2.circle(img2, tuple([pt2[0],pt2[1]]), 5, color, -1)
    return img1, img2


# draw epipolar lines using matplotlib
def draw_extreme_lines_plt(img1, img2, lines1, lines2, epipole1, epipole2):
    ''' img1 - image on which we draw the epilines for the points in img2
        pts1 - points on img1
        lines - corresponding epilines '''
    height, width, color_channel_number = img1.shape

    expand_u_left1 = min(0,epipole1[0])
    expand_u_right1 = max(width, epipole1[0])
    expand_u_left2 = min(0,epipole2[0])
    expand_u_right2 = max(width, epipole2[0])

    plt.subplot(121)
    plt.imshow(img1)
    cnt = 0
    for line1 in lines1:
        x0, y0 = map(int, [expand_u_left1, -(line1[2]+line1[0]*expand_u_left1) / line1[1]])
        x1, y1 = map(int, [expand_u_right1, -(line1[2] + line1[0] * expand_u_right1) / line1[1]])
        if cnt<2:
            plt.plot((x0,x1),(y0,y1), color='blue', linewidth=1)
            cnt+=1
        else:
            plt.plot((x0,x1),(y0,y1), color='red', linewidth=1)
    plt.xlim(0,1000), plt.ylim(2000, -200)  # todo: change the limit to adapt to all conditions.

    plt.subplot(122)
    plt.imshow(img2)
    cnt = 0
    for line2 in lines2:
        x2, y2 = map(int, [expand_u_left2, -(line2[2] + line2[0] * expand_u_left2) / line2[1]])
        x3, y3 = map(int, [expand_u_right2, -(line2[2] + line2[0] * expand_u_right2) / line2[1]])
        if cnt<2:
            plt.plot((x2,x3),(y2,y3),color='r', linewidth=1)
            cnt+=1
        else:
            plt.plot((x2,x3),(y2,y3), color='b', linewidth=1)

    plt.xlim(0,1000), plt.ylim(2000, -200)
    plt.show()


def find_extreme_epilines(img, epipole):
    height, width, color_channel = img.shape

    x1 = epipole[0]
    y1 = epipole[1]

    a = np.array([0, 0, 1])
    b = np.array([width, 0, 1])
    c = np.array([width, height, 1])
    d = np.array([0, height, 1])

    vertex = []

    if x1 < 0:
        if y1 < 0:
            vertex = [d, b]
        elif 0 <= y1 <= height:
            vertex = [d, a]
        elif y1 > height:
            vertex = [c, a]
    elif 0 <= x1 <= width:
        if y1 < 0:
            vertex = [a, b]
        elif 0 <= y1 <= height:
            vertex = [d, d]  # todo: probably not right
        elif y1 > height:
            vertex = [c, d]
    elif x1 > width:
        if y1 < 0:
            vertex = [a, c]
        elif 0 <= y1 <= height:
            vertex = [b, c]
        elif y1 > height:
            vertex = [b, d]

    x2 = vertex[0][0]
    y2 = vertex[0][1]
    x3 = vertex[1][0]
    y3 = vertex[1][1]

    extreme_epiline1 = np.array([1 / (x2 - x1), -1 / (y2 - y1), y1 / (y2 - y1) - x1 / (x2 - x1)])
    extreme_epiline2 = np.array([1 / (x3 - x1), -1 / (y3 - y1), y1 / (y3 - y1) - x1 / (x3 - x1)])

    return extreme_epiline1, extreme_epiline2, vertex


def polar_rectification(img1, img2, matrix_f, epipole1, epipole2):
    # determining the common region

    # find extreme epipolar lines
    extreme_epiline11, extreme_epiline12, vertex1 = find_extreme_epilines(img1, epipole1)
    extreme_epiline21, extreme_epiline22, vertex2 = find_extreme_epilines(img2, epipole2)

    # transfer the extreme epipolar lines in image2 to image1
    extreme_epiline13 = matrix_f.T @ vertex2[0]
    extreme_epiline14 = matrix_f.T @ vertex2[1]

    # transfer the extreme epipolar lines in image1 to image2
    extreme_epiline23 = matrix_f @ vertex1[0]
    extreme_epiline24 = matrix_f @ vertex1[1]

    extreme_epilines1 = np.array([extreme_epiline11, extreme_epiline12, extreme_epiline13, extreme_epiline14])
    extreme_epilines2 = np.array([extreme_epiline21, extreme_epiline22, extreme_epiline23, extreme_epiline24])

    img1, img2 = draw_lines_cv2(img1, img2, extreme_epilines1, extreme_epilines2, np.array(vertex1), np.array(vertex2))
    draw_extreme_lines_plt(img1, img2, extreme_epilines1, extreme_epilines2, epipole1, epipole2)

    # common region(choose two epipolar lines as boundry, Figure 4)
    if extreme_epilines1[2].dot(vertex1[0]) * extreme_epilines1[2].dot(vertex1[1]) < 0:
        region_start_epl1 = extreme_epilines1[2]
    else:
        region_start_epl1 = extreme_epilines1[0]

    if extreme_epilines1[3].dot(vertex1[0]) * extreme_epilines1[3].dot(vertex1[1]) < 0:
        region_end_epl1 = extreme_epilines1[3]
    else:
        region_end_epl1 = extreme_epilines1[1]

    # reconstruction





def main():
    # read images
    img1 = cv2.imread('../data/04L5m2.jpg')  # left image
    b1, g1, r1 = cv2.split(img1)
    img1 = cv2.merge([r1, g1, b1])
    img2 = cv2.imread('../data/04R5m2.jpg')  # right image
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

    # calculates epipolar lines
    epilines1 = compute_correspond_epilines(pts2_h, which_image=2, matrix_f=matrix_f)
    epilines2 = compute_correspond_epilines(pts1_h, which_image=1, matrix_f=matrix_f)

    # # draw epipolar lines and corresponding points
    # img1, img2 = draw_lines(img1, img2, epilines1, epilines2, pts1_nh, pts2_nh)
    #
    # plt.subplot(121), plt.imshow(img1)
    # plt.subplot(122), plt.imshow(img2)
    # plt.show()

    polar_rectification(img1, img2, matrix_f, epipole1, epipole2)

    print('debug')


if __name__ == '__main__':
    main()

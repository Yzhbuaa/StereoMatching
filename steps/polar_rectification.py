#  implementation of Marc Pollefeys's polar rectification algorithm)

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


def create_blank_image(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((int(height), int(width), 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def interpolate_pixels_along_line(x0, y0, x1, y1):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(int(xpxl0 + 1), int(xpxl1)):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels


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
        img1 = cv2.circle(img1, tuple([pt1[0], pt1[1]]), 5, color, -1)

        x2, y2 = map(int, [0, -line2[2] / line2[1]])
        x3, y3 = map(int, [width, -(line2[2] + line2[0] * width) / line2[1]])
        img2 = cv2.line(img2, (x2, y2), (x3, y3), color, 2)
        img2 = cv2.circle(img2, tuple([pt2[0], pt2[1]]), 5, color, -1)
    return img1, img2


# draw epipolar lines using matplotlib
def draw_extreme_lines_plt(img1, img2, lines1, lines2, epipole1, epipole2):
    ''' img1 - image on which we draw the epilines for the points in img2
        pts1 - points on img1
        lines - corresponding epilines '''
    height, width, color_channel_number = img1.shape

    expand_u_left1 = min(0, epipole1[0])
    expand_u_right1 = max(width, epipole1[0])
    expand_u_left2 = min(0, epipole2[0])
    expand_u_right2 = max(width, epipole2[0])

    plt.subplot(121)
    plt.imshow(img1)
    cnt = 0
    for line1 in lines1:
        x0, y0 = map(int, [expand_u_left1, -(line1[2] + line1[0] * expand_u_left1) / line1[1]])
        x1, y1 = map(int, [expand_u_right1, -(line1[2] + line1[0] * expand_u_right1) / line1[1]])
        if cnt < 2:
            plt.plot((x0, x1), (y0, y1), color='blue', linewidth=1)
            cnt += 1
        else:
            plt.plot((x0, x1), (y0, y1), color='red', linewidth=1)
    plt.xlim(0, 1000), plt.ylim(2000, -200)  # todo: change the limit to adapt to all conditions.

    plt.subplot(122)
    plt.imshow(img2)
    cnt = 0
    for line2 in lines2:
        x2, y2 = map(int, [expand_u_left2, -(line2[2] + line2[0] * expand_u_left2) / line2[1]])
        x3, y3 = map(int, [expand_u_right2, -(line2[2] + line2[0] * expand_u_right2) / line2[1]])
        if cnt < 2:
            plt.plot((x2, x3), (y2, y3), color='r', linewidth=1)
            cnt += 1
        else:
            plt.plot((x2, x3), (y2, y3), color='b', linewidth=1)

    plt.xlim(0, 1000), plt.ylim(2000, -200)
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
    region = 1  # Figure 3

    if x1 < 0:
        if y1 < 0:
            vertex = [d, b]
            region = 1
        elif 0 <= y1 <= height:
            vertex = [d, a]
            region = 4
        elif y1 > height:
            vertex = [c, a]
            region = 7
    elif 0 <= x1 <= width:
        if y1 < 0:
            vertex = [a, b]
            region = 2
        elif 0 <= y1 <= height:
            vertex = [d, d]  # todo: probably not right
            region = 5
        elif y1 > height:
            vertex = [c, d]
            region = 8
    elif x1 > width:
        if y1 < 0:
            vertex = [a, c]
            region = 3
        elif 0 <= y1 <= height:
            vertex = [b, c]
            region = 6
        elif y1 > height:
            vertex = [b, d]
            region = 9

    x2 = vertex[0][0]
    y2 = vertex[0][1]
    x3 = vertex[1][0]
    y3 = vertex[1][1]

    extreme_epiline1 = np.array([1 / (x2 - x1), -1 / (y2 - y1), y1 / (y2 - y1) - x1 / (x2 - x1)])
    extreme_epiline2 = np.array([1 / (x3 - x1), -1 / (y3 - y1), y1 / (y3 - y1) - x1 / (x3 - x1)])

    return extreme_epiline1, extreme_epiline2, vertex, region


def compute_common_region(extreme_epilines, vertex):
    if extreme_epilines[2].dot(vertex[0]) * extreme_epilines[2].dot(vertex[1]) < 0:
        start_epl = extreme_epilines[2]
    else:
        start_epl = extreme_epilines[0]

    if extreme_epilines[3].dot(vertex[0]) * extreme_epilines[3].dot(vertex[1]) < 0:
        end_epl = extreme_epilines[3]
    else:
        end_epl = extreme_epilines[1]
    return start_epl, end_epl


'''compute the epipolar lines used to reconstruct the image'''


def compute_epilines_list(epp, start_epl, end_epl):
    epp_nh = np.array([epp[0], epp[1]])  # non_homogeneous epipole1's coordinates

    current_point = np.array([0, -(start_epl[2] / start_epl[1])])
    end_point = np.array([0, -(end_epl[2] / end_epl[1])])

    point_list = []
    epl_list = []
    while current_point[1] < end_point[1]:
        point_list.append(current_point)
        step = np.linalg.norm(epp_nh - current_point) / epp_nh[0]
        next_point = current_point + np.array([0, step])
        epl = np.cross(np.append(current_point, 1), epp)
        epl_list.append(epl / epl[-1])
        current_point = next_point

    return point_list, epl_list


# generate the rectified image
# Return:
#   rectified_img
#   first_last_pixels_distance_list, record the distance of the first and the last pixels
def generate_rectified_image(pt_list, epipole, img, rectified_img, angle_list, max_height, max_biggest_angle):
    # angle_list = []
    # for pt1, pt2 in zip(pt_list, pt_list[1:]):
    #     angle = 0
    #     if (pt1[1] - epipole[1]) * (pt2[1] - epipole[1]) >= 0:
    #         angle = abs(np.arctan(np.abs(epipole[1] - pt2[1]) / epipole[0]) - np.arctan(
    #             np.abs(epipole[1] - pt1[1]) / epipole[0]))
    #     else:
    #         angle = abs(np.arctan(np.abs(epipole[1] - pt2[1]) / epipole[0]) + np.arctan(
    #             np.abs(epipole[1] - pt1[1]) / epipole[0]))
    #     angle_list.append(angle)

    # white = (255,255,255)
    # r_max = np.linalg.norm(np.array([0, img.shape[0]]) - np.array([epipole[0], epipole[1]]))
    # r_min = epipole[0] - img.shape[1]
    # biggest_angle = sum(angle_list)
    # rectified_img = create_blank_image(r_max, biggest_angle * (r_max-r_min) * img.shape[0] / img.shape[1] + 1, rgb_color=white)

    first_last_pixels_distance_list = []
    current_angle = 0
    theta_list = []
    for pt, angle in zip(pt_list, angle_list[:-1]):
        pixels = interpolate_pixels_along_line(epipole[0], epipole[1], pt[0], pt[1])
        valid_pixels = np.array([x for x in pixels if img.shape[1] > x[0] >= 0 and img.shape[0] > x[1] >= 0])
        if valid_pixels.size != 0:
            first_last_pixels_distance_list.append(np.linalg.norm(valid_pixels[0] - valid_pixels[-1]))

        for px in valid_pixels:
            r = np.linalg.norm(px - np.array([epipole[0], epipole[1]]))
            theta = int(current_angle / max_biggest_angle * max_height)
            rectified_img[theta, int(r)] = img[int(px[1]), int(px[0])]
            theta_list.append(theta)
            print(int(current_angle / max_biggest_angle * max_height))
        current_angle = current_angle + angle

    return rectified_img, first_last_pixels_distance_list, theta_list


def polar_rectification(img1, img2, matrix_f, epipole1, epipole2):
    # determining the common region

    # find extreme epipolar lines
    extreme_epiline11, extreme_epiline12, vertex1, region1 = find_extreme_epilines(img1, epipole1)
    extreme_epiline21, extreme_epiline22, vertex2, region2 = find_extreme_epilines(img2, epipole2)

    # transfer the extreme epipolar lines in image2 to image1
    extreme_epiline13 = matrix_f.T @ vertex2[0]
    extreme_epiline14 = matrix_f.T @ vertex2[1]

    # transfer the extreme epipolar lines in image1 to image2
    extreme_epiline23 = matrix_f @ vertex1[0]
    extreme_epiline24 = matrix_f @ vertex1[1]

    extreme_epilines1 = np.array([extreme_epiline11, extreme_epiline12, extreme_epiline13, extreme_epiline14])
    extreme_epilines2 = np.array([extreme_epiline21, extreme_epiline22, extreme_epiline23, extreme_epiline24])

    # img1, img2 = draw_lines_cv2(img1, img2, extreme_epilines1, extreme_epilines2, np.array(vertex1), np.array(vertex2))
    draw_extreme_lines_plt(img1, img2, extreme_epilines1, extreme_epilines2, epipole1, epipole2)

    # common region(choose two epipolar lines as boundry, Figure 4)
    start_epl1, end_epl1 = compute_common_region(extreme_epilines1, vertex1)
    start_epl2, end_epl2 = compute_common_region(extreme_epilines2, vertex2)

    # find the border opposite to the epipole1 todo: judgement refinement
    # i.e. the border furthest away from epipole1
    # in my case, the border is u = 0
    if region1 == 6:
        opposite_border = np.array([1, 0, 0])  # function of the opposite border

    # todo: generalization
    # points and epipolar lines used to reconstruct the image
    pt_list1, epl_list1 = compute_epilines_list(epipole1, start_epl1, end_epl1)
    pt_list2, epl_list2 = compute_epilines_list(epipole2, start_epl2, end_epl2)

    epl_list221 = []  # epipolar lines transferred back to the first image
    pt_list221 = []
    for i in pt_list2:
        epl221 = matrix_f.T @ np.append(i, 1)
        pt221 = np.array([0, -epl221[2] / epl221[1]])
        epl_list221.append(epl221)
        pt_list221.append(pt221)

    pt_list = []  # the final pt_list used to reconstruct the image(in image1)
    epl_list = []
    for pt1, pt221, e1, e221 in zip(pt_list1, pt_list221, epl_list1, epl_list221):
        if pt1[1] <= pt221[1]:
            pt_list.append(pt221)
            epl_list.append(e221)
        else:
            pt_list.append(pt1)
            epl_list.append(e1)

    angle_list = []
    angle = 0
    for pt1, pt2 in zip(pt_list, pt_list[1:]):
        if (pt1[1] - epipole1[1]) * (pt2[1] - epipole1[1]) >= 0:
            angle = abs(np.arctan(np.abs(epipole1[1] - pt2[1]) / epipole1[0]) - np.arctan(
                np.abs(epipole1[1] - pt1[1]) / epipole1[0]))
        else:
            angle = abs(np.arctan(np.abs(epipole1[1] - pt2[1]) / epipole1[0]) + np.arctan(
                np.abs(epipole1[1] - pt1[1]) / epipole1[0]))
        angle_list.append(angle)

    white = (255, 255, 255)
    r_max = np.linalg.norm(np.array([0, img1.shape[0]]) - np.array([epipole1[0], epipole1[1]]))
    r_min = epipole1[0] - img1.shape[1]
    biggest_angle = sum(angle_list)
    width = r_max
    height = biggest_angle * (r_max - r_min) * img1.shape[0] / img1.shape[1] + 1

    # reconstruct image2
    # a = np.array([[0.01,0.01,0.01]])
    # print(np.reshape(epipole2,(3,1)))
    # homography = np.cross(epipole2, matrix_f) + np.reshape(epipole2, (3,1)) @ a
    # epl_list_prime = homography

    # find the corredponding epilines and points in image2 used to recontruct image2
    epl_list_prime = []
    pt_list_prime = []
    for pt in pt_list:
        epl_prime = matrix_f @ np.append(pt, 1)
        epl_list_prime.append(epl_prime)
        pt_prime = np.array([0, -epl_prime[2] / epl_prime[1]])
        pt_list_prime.append(pt_prime)

    angle_list_prime = []
    angle = 0
    for pt1, pt2 in zip(pt_list_prime, pt_list_prime[1:]):
        if (pt1[1] - epipole2[1]) * (pt2[1] - epipole2[1]) >= 0:
            angle = abs(np.arctan(np.abs(epipole2[1] - pt2[1]) / epipole2[0]) - np.arctan(
                np.abs(epipole2[1] - pt1[1]) / epipole2[0]))
        else:
            angle = abs(np.arctan(np.abs(epipole2[1] - pt2[1]) / epipole2[0]) + np.arctan(
                np.abs(epipole2[1] - pt1[1]) / epipole2[0]))
        angle_list_prime.append(angle)

    r_max_prime = np.linalg.norm(np.array([0, img2.shape[0]]) - np.array([epipole2[0], epipole2[1]]))
    r_min_prime = epipole2[0] - img2.shape[1]
    biggest_angle_prime = sum(angle_list_prime)
    width_prime = r_max_prime
    height_prime = biggest_angle_prime * (r_max_prime - r_min_prime) * img2.shape[0] / img2.shape[1] + 1

    rectified_img = create_blank_image(max(width, width_prime), max(height, height_prime) + 1, rgb_color=white)
    rectified_img_prime = create_blank_image(max(width, width_prime), max(height, height_prime) + 1, rgb_color=white)

    rectified_image, first_last_pixels_distance_list, theta_list = generate_rectified_image(pt_list, epipole1, img1,
                                                                                            rectified_img, angle_list,
                                                                                            max(height, height_prime),
                                                                                            max(biggest_angle,
                                                                                                biggest_angle_prime))
    rectified_image = cv2.flip(rectified_image, 1)

    rectified_img_prime, first_last_pixels_distance_list_prime, theta_list_prime = generate_rectified_image(
        pt_list_prime, epipole2, img2, rectified_img_prime, angle_list, max(height, height_prime),
        max(biggest_angle, biggest_angle_prime))
    rectified_img_prime = cv2.flip(rectified_img_prime, 1)

    cv2.imwrite(filename='../data/rectified_img.jpg', img=rectified_image)
    cv2.imwrite(filename='../data/rectified_img_prime.jpg', img=rectified_img_prime)

    with open('../data/theta_list.txt', 'w') as f:
        for i in theta_list:
            f.write("%s\n" % i)

    with open('../data/theta_list_prime.txt', 'w') as f:
        for i in theta_list_prime:
            f.write("%s\n" % i)

    plt.subplot(121)
    plt.imshow(rectified_image)

    plt.subplot(122)
    plt.imshow(rectified_img_prime)

    plt.show()

    print('debug')


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

    # draw epipolar lines and corresponding points
    # img1, img2 = draw_lines_cv2(img1, img2, epilines1, epilines2, pts1_nh, pts2_nh)

    # cv2.imwrite('../data/img1_with_8epipolar_lines.jpg',img1)
    # cv2.imwrite('../data/img2_with_8epipolar_lines.jpg',img2)

    # plt.subplot(121), plt.imshow(img1)
    # plt.subplot(122), plt.imshow(img2)
    # plt.show()

    polar_rectification(img1, img2, matrix_f, epipole1, epipole2)

    print('debug')


if __name__ == '__main__':
    main()

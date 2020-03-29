from matplotlib import pyplot as plt
import numpy as np
import cv2

# this is just to unconfuse pycharm
try:
    from cv2 import cv2
except ImportError:
    pass


''' sum of squared difference algorithm 
    :parameter
        big_img - image contains two rectified images
        pt - point in the left part of the big_img
    :return
        pt_r - the most appropriate point calculated by sum of 
            squared difference.'''
def ssd(rectified_img1, big_img, pt, window_size):
    pt = np.int_(pt)
    line = pt[1]
    sum_list = []
    for i in range(rectified_img1.shape[1], big_img.shape[1] - int(window_size / 2)):
        sum = 0
        searching_pt = np.array([i, line])
        for j in range(pt[0] - int(window_size / 2), pt[0] + int(window_size / 2)):
            for k in range(pt[1] - int(window_size / 2), pt[1] + int(window_size / 2)):
                temp = int((int(big_img[k][j]) - int(big_img[k][j + i - pt[0]])) ** 2)
                sum = sum + temp
        sum_list.append(sum)

    min_sum = min(sum_list)
    min_sum_index = sum_list.index(min(sum_list))
    searched_pt = np.array([min_sum_index + rectified_img1.shape[1], line])
    return searched_pt, min_sum


def main():
    rectified_img1 = cv2.imread('../data/img1_planar_rectified_interpolated.jpg', cv2.IMREAD_GRAYSCALE)
    rectified_img2 = cv2.imread('../data/img2_planar_rectified_interpolated.jpg', cv2.IMREAD_GRAYSCALE)

    # plt.subplot(121)
    # plt.imshow(rectified_img1, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(rectified_img2, cmap='gray')
    # plt.show()
    # big_img = np.concatenate((rectified_img1, rectified_img2), axis=1)
    # cv2.imwrite('../data/big_img_planar_rectified_interpolated.jpg',big_img)
    # plt.imshow(big_img)
    # plt.show()

    pts = np.loadtxt('../data/ssd_points_planar_rectified_interpolated.txt')

    for i in range(5, 52, 2):
        big_img = np.concatenate((rectified_img1, rectified_img2), axis=1)
        window_size = i  # width = height = window_size

        searched_pts = []
        min_sums = []
        pts = np.int_(pts)
        for pt in pts:
            searched_pt, min_sum = ssd(rectified_img1, big_img, pt, window_size)
            searched_pts.append(searched_pt)
            min_sums.append(min_sum)

        for pt, spt in zip(pts, searched_pts):
            color = 0
            big_img = cv2.circle(big_img, tuple([pt[0], pt[1]]), 10, color, -1)
            big_img = cv2.circle(big_img, tuple([spt[0], spt[1]]), 10, color, -1)
            big_img = cv2.line(big_img,tuple([pt[0], pt[1]]),tuple([spt[0], spt[1]]),color,2)

        file_name = '../data/planar_rectificated_interpolated_search_window_size__' + str(window_size) + '.jpg'
        cv2.imwrite(file_name, big_img)
        plt.imshow(big_img, cmap='gray')
        plt.show()

    print('debug')


if __name__ == '__main__':
    main()

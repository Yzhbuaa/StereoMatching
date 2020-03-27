import numpy as np
from steps.file_parser import parse_data
from utils.timer import timer
from utils.coordinate_transform import homogeneous_coordinate_calc


def normalization(points):
    end = timer()

    u_x = points[:, 0].mean()  # Centroid
    u_y = points[:, 1].mean()

    x_shifted = points[:, 0] - u_x  # shift origin to centroid
    y_shifted = points[:, 1] - u_y

    average_distance_from_origin = np.sqrt(x_shifted**2+y_shifted**2).mean()

    scale = np.sqrt(2)/average_distance_from_origin

    end("Eight points algorithm, input normalization")

    return np.array([
        [scale,   0,   -scale * u_x],
        [0,   scale,   -scale * u_y],
        [0,       0,              1]
    ])


# implementation of eight points algorithms
# input: 2D numpy array, shape = {16,2}, stores the left image's points and
#        the right image's points. Attention: stores in an alternative pattern!
# return: fundamental matrix in
def eight_points_algorithm(sixteen_points):

    # slices image points in left and right images
    left_img_points = sixteen_points[0:16:2]
    right_img_points = sixteen_points[1:16:2]

    # Calculates the normalization matrix
    left_img_normalization_matrix = normalization(left_img_points)
    right_img_normalization_matrix = normalization(right_img_points)

    matrix_a = []  # original matrix A in Af = 0
    normalized_matrix_a = []  # normalized_matrix_a is the matrix A in equations Af = 0.

    for i in range(0, 8):

        # takes single point(non-homogeneous coordinates)
        left_img_point = left_img_points[i]
        right_img_point = right_img_points[i]

        # turns the single point(non-homogeneous coordinates) into homogeneous coordinates
        left_img_point = homogeneous_coordinate_calc(left_img_point)
        right_img_point = homogeneous_coordinate_calc(right_img_point)

        # binds new names to left_img_point and right_img_point to avoid lines being too long
        x = left_img_point
        x_ = right_img_point

        # matrix_a in list form
        matrix_a.append(np.array([
            x_[0]*x[0], x_[0]*x[1], x_[0], x_[1]*x[0], x_[1]*x[1], x_[1], x[0], x[1], 1
        ]))

        # performs normalization
        normalized_left_img_point = left_img_normalization_matrix @ left_img_point
        normalized_right_img_point = right_img_normalization_matrix @ right_img_point

        # binds new names to normalized_left_img_point and normalized_right_img_point to avoid lines being too long
        x = normalized_left_img_point
        x_ = normalized_right_img_point

        # normalized_matrix_a in list form
        normalized_matrix_a.append(np.array([
            x_[0]*x[0], x_[0]*x[1], x_[0], x_[1]*x[0], x_[1]*x[1], x_[1], x[0], x[1], 1
        ]))

    matrix_a = np.array(matrix_a)
    normalized_matrix_a = np.array(normalized_matrix_a)

    # Linear solution
    u, s, vh = np.linalg.svd(normalized_matrix_a)
    array_f = vh[-1]
    print(normalized_matrix_a @ array_f)

    # Constraint enforcement
    matrix_f = array_f.reshape((3, 3))
    u1, s1, vh1 = np.linalg.svd(matrix_f)
    s1[-1] = 0
    matrix_f = (u1 @ np.diag(s1)) @ vh1

    t = left_img_normalization_matrix
    t_ = right_img_normalization_matrix

    # denormalization
    matrix_f = t_.T @ matrix_f @ t

    matrix_f = matrix_f / matrix_f[-1][-1]

    return matrix_f


def main():
    sixteen_points = parse_data(file_name='eight_points_qu')
    matrix_f = eight_points_algorithm(sixteen_points)
    print('F:')
    print(matrix_f)


if __name__ == '__main__':
    main()


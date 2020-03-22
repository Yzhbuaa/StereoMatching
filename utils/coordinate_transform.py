import numpy as np


# add 1 to nonhomogeneous coordinates in order to make it a homogeneous coordinates.
# returns nonhomogeneous_coordinates corresponding homogeneous_coordinates.
# input: 2-D array, shape[0] =  2, shape[1] =  1
# output: 2-D array, shape[0] = 3, shape[1] = 1
def homogeneous_coordinate_calc(nonhomogeneous_coordinate):

    homogeneous_coordinate = np.array([
        nonhomogeneous_coordinate[0],
        nonhomogeneous_coordinate[1],
        1
    ])

    return homogeneous_coordinate

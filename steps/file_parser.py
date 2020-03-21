import numpy as np


def parse_data(base_path="data/", file_name="eight_points_data", ext=".txt"):
    path = base_path + file_name + ext

    eight_points = np.loadtxt(base_path + file_name + ext)
    sixteen_points = eight_points.reshape((eight_points.shape[0]*2, 2))

    return sixteen_points

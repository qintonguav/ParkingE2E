import numpy as np
from scipy.special import comb


def bezier_curve(points, num_points, item_number):
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, item_number))
    for i in range(num_points):
        for j in range(n + 1):
            curve[i] += comb(n, j) * (1 - t[i])**(n - j) * t[i]**j * points[j]
    return curve

def fitting_curve(raw_data, num_points, item_number):
    control_points = np.array(raw_data)
    smoothed_curve = bezier_curve(control_points, num_points, item_number)
    return smoothed_curve.tolist()

def calculate_tangent(points, mode):
    num_points = len(points)

    if num_points < 2:
        return [0.]

    tangent = np.zeros((num_points, 2))
    for i in range(num_points):
        if mode == "three_point":
            if i == 0:
                tangent[i] = -(points[i+1] - points[i])
            elif i == num_points - 1:
                tangent[i] = -(points[i] - points[i-1])
            else:
                tangent[i] = -(points[i+1] - points[i-1])
        elif mode == "five_point":
            if i == 0:
                tangent[i] = -(points[i+1] - points[i])
            elif i == num_points - 1:
                tangent[i] = -(points[i] - points[i-1])
            elif (i == 1) or (i == num_points - 2):
                tangent[i] = -(points[i] - points[i-1])
            else:
                tangent[i] = -((points[i - 2] - 8 * points[i - 1] + 8*points[i + 1] - points[i + 2]) / 12)
        elif mode == "three_point_back":
            if i == 0:
                tangent[i] = -(points[i+1] - points[i])
            else:
                tangent[i] = -(points[i] - points[i-1])
        elif mode == "three_point_front":
            if i == num_points - 1:
                tangent[i] = -(points[i] - points[i-1])
            else:
                tangent[i] = -(points[i+1] - points[i])
        else:
            assert print("Error mode!")

    traj_heading_list = np.rad2deg(np.arctan2(tangent[:, 1], tangent[:, 0])).tolist()
    
    if np.any(np.isnan(traj_heading_list)):
        print("NaN")
    return traj_heading_list


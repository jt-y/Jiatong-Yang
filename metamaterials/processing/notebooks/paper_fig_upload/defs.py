import os
import pickle
import glob
import skimage
import skimage.transform
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from os import listdir
from os.path import isfile, join
import shutil, os

import sys

sys.path.append('../../functions')

import chirp_functions as proc
import stmpy



def return_sublat_triangle(origin, coord_crt, lat_vecs, sub_lat_vecs):
    "Enter the coordinates of the atomic site in the unit cell and the origin of the lattice (sublattice 1)."
    "Return the sublattice that the input coordinate belongs to."
    "Only works for triangle lattice. The first lattice vector is the horizontal one, the second is the non-horizontal one."

    # Shift the coordinate to the origin
    rel_coord = np.array(coord_crt) - np.array(origin)

    n2 = rel_coord[1] // lat_vecs[1][1]
    n1 = (rel_coord[0] - n2 * lat_vecs[1][0]) // lat_vecs[0][0]

    frac = rel_coord - n1 * lat_vecs[0] - n2 * lat_vecs[1]

    if np.allclose(frac, sub_lat_vecs[0], atol=1e-1):
        return 0
    elif np.allclose(frac, sub_lat_vecs[1], atol=1):
        return 1
    elif np.allclose(frac, sub_lat_vecs[2], atol=1):
        return 2
    else:
        return -1
    

def generate_k_space_sweep(pt_list, num_points):
    """
    Generate a sweep of k-points along the specified path in reciprocal space.

    Parameters:
    pt_list (np.array): List of points in reciprocal space to define the path.
    num_points (int): Total number of k-points to generate along the path.

    Returns:
    b (np.array): 1D array that contains the cumulative distance along the path for each k-point.
    kx_sweep (np.array): 1D array of kx coordinates for the sweep.
    ky_sweep (np.array): 1D array of ky coordinates for the sweep.
    """

    # First calculate the total length of the path in reciprocal space
    total_length = 0
    for i in range(len(pt_list) - 1):
        total_length += np.linalg.norm(pt_list[i+1] - pt_list[i])

    kx_sweep = np.zeros(0)
    ky_sweep = np.zeros(0)
    b = np.zeros(0)
    
    # Now generate the k-points along the path
    for i in range(len(pt_list) - 1):
        start_pt = pt_list[i]
        end_pt = pt_list[i+1]
        segment_length = np.linalg.norm(end_pt - start_pt)
        

        # Generate k-points for this segment
        if i == len(pt_list) - 2:  # Ensure the last point is included in the final segment
            num_points_segment = int(np.ceil(num_points * (segment_length / total_length)))
            kx_segment = np.linspace(start_pt[0], end_pt[0], num_points_segment, endpoint=True)
            ky_segment = np.linspace(start_pt[1], end_pt[1], num_points_segment, endpoint=True)
            b_segment = np.linspace(total_length * i / (len(pt_list) - 1), total_length * (i + 1) / (len(pt_list) - 1), num_points_segment, endpoint=True)
        else:
            num_points_segment = int(np.round(num_points * (segment_length / total_length)))
            kx_segment = np.linspace(start_pt[0], end_pt[0], num_points_segment, endpoint=False)
            ky_segment = np.linspace(start_pt[1], end_pt[1], num_points_segment, endpoint=False)
            b_segment = np.linspace(total_length * i / (len(pt_list) - 1), total_length * (i + 1) / (len(pt_list) - 1), num_points_segment, endpoint=False)

        b = np.hstack((b, b_segment))
        kx_sweep = np.hstack((kx_sweep, kx_segment))
        ky_sweep = np.hstack((ky_sweep, ky_segment))


    return b, kx_sweep, ky_sweep


def visualize_sweep(k_sweep, BZ_points):
    """
    Plot the k-space sweep path along with the first Brillouin zone points.
    """

    plt.figure(figsize=(6, 4))
    plt.axis('equal')
    plt.plot(BZ_points[:, 0], BZ_points[:, 1], '-', label='BZ Points')
    plt.plot(k_sweep[0], k_sweep[1], 'o', label='k-space Sweep')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title('k-space Sweep Path')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

    return 


def rotate_points(points, angles):
    """
    Rotate one point in 2D by an array of angles.

    Parameters:
    points (np.array): 1D array of length 2 containing the x and y coordinates of the point to be rotated. 
    angles (np.array): 1D array of rotation angles in radians.

    Returns:
    np.array: 2D array of shape (len(angles), 2) containing the rotated coordinates for each angle.
    """

    # Create the rotation matrix for each angle
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)
    rotation_matrices = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]).transpose(2, 0, 1)

    # Rotate the point using the rotation matrices
    rotated_points = np.einsum('ijk,j->ik', rotation_matrices, points)

    return rotated_points
    
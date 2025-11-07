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
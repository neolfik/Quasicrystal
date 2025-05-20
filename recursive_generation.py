import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from utilities import (
    deduplicate as dp,
    rotate_points as rp
)

## This file includes the generation of points via recursion, this method is faster than the procedural generation used in previous version

## Adds one unit to a specified center using rotation of points
def single_center_addition(c, a):
    """
    Generate points of the mesh around a center point.
    """
    v = np.sqrt(3) / 2 * a
    s = a / 2
    triangle = np.vstack([
        [0, 0],
        [0, a],
        [s, a + v],
        [v + s, v + s]
    ])
    return rp(triangle) + c

def add_points(a, points):
    """
    Parallelized points adition to speed up recursion.
    """
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(single_center_addition, points, repeat(a)))
    return np.vstack([points] + results)

def quasicrystal(tside, bp, num):
    """
    Recurent function used to generate the quasicrystal
    """
    points = np.array([[0, 0]])
    if num > 0:
        side = tside * (2 + np.sqrt(3)) ** num
        points = quasicrystal(tside, bp, num - 1)
        points = add_points(side, points)
    else:
        points = add_points(tside, bp)
    return dp(points)

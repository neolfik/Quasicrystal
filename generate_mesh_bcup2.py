import numpy as np
from itertools import combinations
from shapely.geometry import Polygon
from shapely.prepared import prep
from concurrent.futures import ProcessPoolExecutor, as_completed
from quasi_tiling import (
    build_points as bp,
    build_points2 as bp2,
    rotated_points as rp,
    sort_by_distance_from_origin as sb,
)
import time

## This code generates the quasicrystal with use of geomtry from quasi_tiling


def dedup_preserve_order(points, decimals=6):
    """
    Deduplicates inserted points with a certain decimal accuracy

    Parameters:
        points (numpy.ndarray) - input points to be deduplicated
        decimals (int) - accuracy based on numbers after decimal point
    """
    _, idx = np.unique(np.round(points, decimals), axis=0, return_index=True)
    return points[np.sort(idx)]

def check_overlap(polygons1, polygons2, expected_area):
    """
    Checks the overlap of polygons from the quasi periodic tiling with respect to allowed overlap

    Parameters:
        polygons1 (numpy.ndarray) - an array of Polygon class objects that are checked but were used from previous steps
        polygons2 (numpy.ndarray) - an array of new Polygon class objects the are checked with poligon1 and then crosschecked
        expected_area (float) - allowed overlap between the polygons
    """
    ea = np.round(expected_area, 3)
    threshold = {0, ea}

    ## map(prep, polygons1) is used to check the overlap much faster lowering the computing time, it is a function of shapely
    for poly1, prepared_poly1 in zip(polygons1, map(prep, polygons1)):
        for poly2 in polygons2:
            if prepared_poly1.intersects(poly2):
                area = np.round(poly1.intersection(poly2).area, 3)
                if area not in threshold:
                    return False

    ## the map optimalization is not used here because of the inherent size of polygons2 and more complicated for loop system
    for poly1, poly2 in combinations(polygons2, 2):
        if poly1.intersects(poly2):
            area = np.round(poly1.intersection(poly2).area, 3)
            if area not in threshold:
                return False

    return True

def wrapper_base(args, use_alt=False):
    """
    Function used for paralelization to generate points of the quasicrystal

    Parameters:
        use_alt (Boolean) - allows the usage of square points generation
        args:
            triplet (numpy.ndarray) - idices of three points used for generation
            temp (numpy.ndarray) - temporary points used to choose the triplet 
            side (float) - lattice parameter
            polygons (numpy.ndarray) - array of Polygon class objects that are used from previous steps
            expected_area (float) - allowed overlap area
    """
    triplet, temp, side, polygons, expected_area = args
    p1, p2, p3 = temp[list(triplet)]

    ## Point generation
    if use_alt:
        pp, peak = bp2(np.vstack((p1, p2, p3)), side)
        if not isinstance(pp, np.ndarray):
            return None
    else:
        for orientation in [1, -1]:
            pp, peak = bp(np.vstack((p1, p2, p3)), side, orientation)
            if isinstance(pp, np.ndarray):
                break
        else:
            return None

    ## Rotation of the points in respect to symetry and making temporary polygons
    pp_all = rp(pp)
    new_polygons = [Polygon(pp_all[i:i+8]) for i in (0, 8, 40)]

    ## Polygon overlap check
    if not check_overlap(polygons, new_polygons, expected_area):
        return None

    ## adjusting new_temporary points to roughly the first quadrant and deleting the peak point from the selection
    new_temp = temp[~np.all(np.abs(temp - peak) < 1e-6, axis=1)]
    mask = (pp_all[:, 0] > 0) & (pp_all[:, 1] >= pp_all[:, 0] / 10) & (pp_all[:, 1] < 10 * pp_all[:, 0])
    new_temp = sb(dedup_preserve_order(np.vstack((new_temp, pp_all[mask])), 6))

    return {
        'pp_all': pp_all,
        'new_temp': new_temp,
        'new_polygons': new_polygons,
        'new_point_count': len(new_temp),
    }

def generate_quasicrystal(cycles, side):
    """
    Manages the generation of the quasicrystal, logs the time spent, the paralelization is made in this function

    Parameters:
        cycles (int) - number of generation cycles
        side (float) - lattice parameter
    """

    start_time = time.time()

    ## The next part is used to plot the starting geometry as it is the same each time
    base_triangle = np.array([[0, 0], [1, 0], [1 + np.sqrt(3) / 2, 0.5]])
    temp, peak = bp(base_triangle, side, 1)
    points = rp(temp)

    polygons = [Polygon(points[i:i+8]) for i in (0, 8, 40)]
    temp = dedup_preserve_order(temp[~np.all(np.abs(temp - peak) < 1e-6, axis=1)], 6)
    temp = sb(temp)
    points = dedup_preserve_order(points, 4)

    expected_area = np.sqrt(3) / 4 * side

    ## Cycles of generation
    for i in range(cycles):
        print(f'\n{"#" * 80}\nCycle {i}')
        combos = list(combinations(range(len(temp)), 3))
        args_list = [(triplet, temp, side, polygons, expected_area) for triplet in combos]

        results = []

        ## Paralelization of the program for faster checking, the library does the thread management itself, is usefull for the loops
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(wrapper_base, args, True) for args in args_list]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                    break

        ## Paralelization for the other point generation
        if not results:
            print('Falling back to standard bp')
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(wrapper_base, args, False) for args in args_list]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                        break

        ## Checks if there are any valid results and updates the used parameters, the code could be cleaned up so that results are not
        # useed because the list is redundant, feel free to do that
        if results:
   
            best_result = results[0]
            new_points = best_result['pp_all']
            new_temp = best_result['new_temp']
            new_polygons = best_result['new_polygons']

            points = np.vstack((points, new_points))
            temp = np.vstack((temp, new_temp))
            polygons.extend(new_polygons)

            points = dedup_preserve_order(points, 4)
            temp = dedup_preserve_order(temp, 6)

            ## clears first few values from the lists, it doesn't affect the generation if it is not higher than 2 and speeds it up
            temp = sb(temp)[2:]

            ## clears the polygons in case the amount gets too big, if you want to generate high number of generation this will
            # this will break the program
            if len(polygons) > 50:
                polygons = polygons[-50:]


            print(f"Added new structure. Total polygons: {len(polygons)}, temp points: {len(temp)}")
        else:
            print(f"No valid combination found in cycle {i}")
            break
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f} seconds.")
        yield i, dedup_preserve_order(points, 4), temp, polygons

def quasicrystal(cycles, side):
    """
    Main call function for quasicrystal generation

    Parameters:
        cycles (int) - number of generation cycles
        side (float) - lattice parameter
    """
    result = None
    for result in generate_quasicrystal(cycles, side):
        pass
    if result:
        _, points, temp, _ = result
        return points, temp
    return None, None


def hexagonal(rows, cols, side):
    """
    Generates points for hexagonal grid

    Parameters:
        rows (int) - number of rows in the generation
        cols (int) - number of columns in the generation
        side (float) - lattice parameter
    """

    half_sqrt3_side = (np.sqrt(3) / 2) * side
    points = []

    for r in range(rows // 2):
        y = r * half_sqrt3_side
        x_offset = side / 2 if r % 2 else 0
        for c in range(cols // 2):
            x = c * side + x_offset

            # Avoid duplicate points at (0,0)
            if x == 0 and y == 0:
                points.append([0, 0])
            else:
                points.extend([
                    [ x,  y],
                    [-x,  y],
                    [ x, -y],
                    [-x, -y]
                ])
    return np.unique(np.array(points),axis = 0)



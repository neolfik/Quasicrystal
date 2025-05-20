import numpy as np
from scipy.ndimage import shift
from concurrent.futures import ProcessPoolExecutor

## In this file provided array of points is transformed into map with gaussians at each input point. Given the number of points, the method
# used evaluates the gaussian on a smaller map around origin and then shifts existing map into position of the input points.
# This method is faster, than computing the additives of each input point, for higher number of points.

## Prepare function for paralelization further in the code
def process_point(args):
    """
    Moves gaussian distribution from the origin to a specified point and returns the shifted map
    """
    px, py, dx, dy, origin_array = args
    shift_y = py / dy
    shift_x = px / dx
    return shift(origin_array, shift=(shift_y, shift_x), order=1, mode='constant')

def add_points(points, sig, N):
    """
    Adds gaussians to each input point with specified standart deviation and a specified accuracy of the map
    """
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    ## Add a 10% buffer to the map to make the resulting plots more user friendly by incorporating the edge of the crystal
    buffer_x = 0.1 * (max_x - min_x) if max_x != min_x else 1.0
    buffer_y = 0.1 * (max_y - min_y) if max_y != min_y else 1.0

    x0, x1 = min_x - buffer_x, max_x + buffer_x
    y0, y1 = min_y - buffer_y, max_y + buffer_y

    difx = x1 - x0
    dify = y1 - y0

    diff = max(abs(difx), abs(dify))
    x = np.linspace(-diff/2, diff/2, 2*N)
    y = np.linspace(-diff/2, diff/2, 2*N)
    xx, yy = np.meshgrid(x, y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    ## Define the gaussian in origin that is to be shifted to the input points
    origin = np.exp(-((xx)**2 + (yy)**2) / (2 * sig**2))
    result = np.zeros_like(origin)


    args_list = [(px, py, dx, dy, origin) for px, py in points]
    
    ## For each point shift and add the gaussian from the origin to specified position in respect to the indexing of the map
    with ProcessPoolExecutor() as executor:
        for shifted in executor.map(process_point, args_list):
            result += shifted
    
    ## Multiply the resulting map by a constant to normalize the values
    result = (1 / (2 * np.pi * sig**2)) * result  

    ## Make a corrected map by multiplying the whole map with gaussian using standart deviation equal to 2/9 of radius. This ensures that
    # three time the standart deviation is in 2/3 of the radius, and thus correction has the desired effect. The correction should supress 
    # the lower maxima caused by the fact, that the used lattice is finite.
    result2 = result * np.exp(-(xx**2+yy**2)/(8*diff**2/81)) * (1 / (8 * np.pi * diff**2/81))
    print(f'Gaussians added for parameters sigma = {sig} and N = {N}')
    return result2,result, x, y, (x0, x1, y0, y1)

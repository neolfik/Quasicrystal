import numpy as np

## This part of code is used to add 2D Gaussian distribution to an array of points


def gauss(x, y, p, sig):
    """
    Defines Gaussina distribution

    Parameters:
        x (float) - x coordinate
        y (float) - y coordinate
        p (numpy.ndarray) - origin point
        sig (float) - Broadening of the distribution or the standard deviation
    """
    ## The Gaussian is defined so that its shift can be easily controled by the coordinates of the points
    return 1 / (2 * np.pi * sig**2) * np.exp(-((x - p[0])**2 + (y - p[1])**2) / (2 * sig**2))

def add_points(points, sig, N):
    """
    Adds Gaussians to points on input and generates a map of values

    Parameters:
        points (numpy.ndarray) - input points
        sig (float) - standard deviation of added Gaussians
        N (int) - Number of values in one principal direction of the map
    """
    ## Here based on the scale of the points the program makes an array of points in which the Gaussians are evaluated 
    # with a 10% size buffer on all sides The program may be little faster with smaller buffer since the Gaussians don't 
    # need to be added to the unused area
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    buffer_x = 0.1 * (max_x - min_x) if max_x != min_x else 1.0
    buffer_y = 0.1 * (max_y - min_y) if max_y != min_y else 1.0

    x0, x1 = min_x - buffer_x, max_x + buffer_x
    y0, y1 = min_y - buffer_y, max_y + buffer_y

    x = np.linspace(x0, x1, N)
    y = np.linspace(y0, y1, N)
    xx, yy = np.meshgrid(x, y)

    map = np.zeros_like(xx)

    # Adds a Gaussian for each point and tracks a count of the Gaussians added, since the program from main.py can run quite long
    i = 0
    for p in points:
        i+=100
        map += gauss(xx, yy, p, sig)
        print(f'Adding Gaussians to point mesh: {round(i/points.shape[0],2)} %', end = '\r')
    print(f'Gaussians added for parameters sigma = {sig} and N = {N}')

    return map, x, y, (x0, x1, y0, y1)

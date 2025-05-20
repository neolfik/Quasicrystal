import os
import numpy as np
import matplotlib.pyplot as plt
import recursive_generation as rg
import add_gauss as ag
from utilities import (
    hexagonal,
    plot_fft_and as pf
)


## This file consists of two parts, each separated by exit() function so that they do not run simultaniously due to their complexity
# and time needed for completion

## The first part takes generated points from .txt files in points folder and adds gaussians to the points in a specified radius onto a map 
# 500x500 or other specified accuracy, it also makes a correction of the map via multiplying by a gaussian distribution with 3*sigma = 2/3*radius
# The program then shows the map and does a 2D fourier transform of the map and its correction, plots the FT zoomed to an area with values
# over 5% of the maximal value and saves all the plots.
# Based on the furthest point in the .txt file it makes a hexagonal grid and repeats the process with the points of the hexagonal grid.


## Ensure base saving folder exists
base_folder = 'Saved_figures'
os.makedirs(base_folder, exist_ok=True)

## Settings of evaluated distances or radii, the mean deviation used for the gaussians, accuracy ar size of the map with gaussians and
# the lattice parameter
distances = [5,10,15,20]
sigmas = [0.08,0.15,0.2,0.3]
N = 1000
a = 1

## Load file used to chose the points
txt_file = os.path.join("points", "recursive_2.txt")
points = np.loadtxt(txt_file, delimiter=",")

## Ensure there are no duplicates, even though there should not be any based on the implementation of points generation
points = np.unique(np.round(points, 3), axis=0)
radii = np.linalg.norm(points, axis=1)

for radius in distances:
    ## Select points within the specified distance
    mask = radii <= 2*radius
    selected_points = points[mask]

    ## Generate hexagonal grid for this radius in a square shape, the grid is bigger than the actual radius to ensure all valid points 
    # are present before choosing only those in the appropriate radius.
    H = int(np.round(8 * radius))
    D = int(np.round(6 * radius * np.sqrt(3)))
    hex_points = hexagonal(D, H, a)
    
    ## Select points within a specified distance
    radiih = np.linalg.norm(hex_points, axis=1)
    mask2 = radiih <= 2*radius
    hex_points = hex_points[mask2]

    # Set up directory structure
    radius_folder = os.path.join(base_folder, f"recursive_3_radius_{radius}")
    os.makedirs(radius_folder, exist_ok=True)

    for sigma in sigmas:
        sigma_folder = os.path.join(radius_folder, f"sigma_{sigma}")
        os.makedirs(sigma_folder, exist_ok=True)

        
        # Add gaussians
        map11, map12, x1, y1, extent1 = ag.add_points(hex_points, sigma, N)
        map21, map22, x2, y2, extent2 = ag.add_points(selected_points, sigma, N)

        ## Plots all the maps and saves the plots
        pf(map11, extent1, "Hexagonal_", sigma_folder,sigma,N,radius,cor=True)
        pf(map12, extent1, "Hexagonal_", sigma_folder,sigma,N,radius)
        pf(map21, extent2, "Quasicrystal_", sigma_folder,sigma,N,radius,cor=True)
        pf(map22, extent2, "Quasicrystal_", sigma_folder,sigma,N,radius)


exit()

## Is used for generation of points, put this at the start of the code after importing used packages to activate this part.
# Be mindful when chosing the number of iterations, at 4 the code outputs 1 million points and is 18MB, at 5 the code outputs 12 million
# points and 250MB file. For the evaluation above only 3 iterations are used.

if __name__ == '__main__':
    points = rg.quasicrystal(1,np.array([0,0]),2)
    np.savetxt('points/recursive_2.txt', points, fmt='%.6f', delimiter=',')
    plt.plot(points[:,0],points[:,1],',',color = 'black')
    plt.show()
exit()



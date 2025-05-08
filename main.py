import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
import generate_mesh_bcup2 as gm
import add_gauss as ag

## This file consists of two parts, each separated by exit() function so that they do not run simultaniously due to their complexity
# and time needed for completion

## The first part takes generated points from .txt files in points folder and adds gaussians to the points onto a map 1000x1000 or other
# specified accuracy. The program then shows the map and does a 2D fourier transform of the map, plots the FT and saves all the plot
# Based on the furthest point in .txt file it makes a hexagonal grid and repeats the porces with the points of hexagonal grid

## minimal time each plot is shown
T = 0.5

## Parameters
sigmas = [0.02, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25,0.3]
Ns = [1000] * len(sigmas)
a = 1

## Ensure base saving folder exists
base_folder = 'Saved_figures'
os.makedirs(base_folder, exist_ok=True)


def dynamic_zoom_region(arr, threshold_ratio=0.1, padding=10):
    ## Zoomes the FT to a range at which the main peaks are realized with a certain threshold ratio relative to the maximum value
    max_val = np.max(arr)
    mask = arr > (threshold_ratio * max_val)
    coords = np.argwhere(mask)

    if coords.size == 0:
        # fallback to center square crop if no peaks found
        center = np.array(arr.shape) // 2
        half_size = min(arr.shape) // 10
        return tuple(slice(center[i] - half_size, center[i] + half_size) for i in range(2))

    top_left = np.maximum(coords.min(axis=0) - padding, 0)
    bottom_right = np.minimum(coords.max(axis=0) + padding, np.array(arr.shape))

    height = bottom_right[0] - top_left[0]
    width = bottom_right[1] - top_left[1]
    side = max(height, width)

    center = (top_left + bottom_right) // 2
    half_side = side // 2

    start = np.maximum(center - half_side, 0)
    end = np.minimum(start + side, np.array(arr.shape))

    start = np.maximum(end - side, 0)

    return tuple(slice(start[i], end[i]) for i in range(2))

## For loop through the .txt files
for txt_file in glob.glob(os.path.join("points","*.txt")):
    points = np.loadtxt(txt_file, delimiter=",")
    points = np.unique(np.round(points, 3), axis=0)

    ## generating the hexagonal grid
    maxdist = int(np.floor(np.max(np.linalg.norm(points, axis=1))))
    H = 2 * maxdist
    D = int(np.floor(4*maxdist/np.sqrt(3)))

    hex_points = gm.hexagonal(D, H, a)

    ## Make folder based in the filename
    file_base = os.path.splitext(txt_file)[0]
    file_folder = os.path.join(base_folder, file_base)
    os.makedirs(file_folder, exist_ok=True)

    ## For starting parameters make the plots
    for sigma, N in zip(sigmas, Ns):
        sigma_folder = os.path.join(file_folder, f"sigma_{sigma}")
        os.makedirs(sigma_folder, exist_ok=True)

        map1, x1, y1, extent1 = ag.add_points(hex_points, sigma, N)
        map2, x2, y2, extent2 = ag.add_points(points, sigma, N)

        plt.figure(figsize=(8, 6))
        plt.imshow(map1, extent=extent1, origin='lower', cmap='hot')
        plt.colorbar(label='Gaussian Intensity')
        plt.title(f"Gaussian Map (Hexagonal grid): σ={sigma}, N={N}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(f"{sigma_folder}/hexmap_sigma{sigma}_N{N}.png")
        plt.pause(T)
        plt.close()

        fft1 = fft2(map1)
        fft1_mag = np.abs(np.fft.fftshift(fft1))
        zoom_slice = dynamic_zoom_region(fft1_mag)
        plt.figure(figsize=(8, 6))
        plt.imshow(fft1_mag[zoom_slice], cmap='viridis')
        plt.title(f"Zoomed FFT (Hexagonal grid): σ={sigma}, N={N}")
        plt.xlabel("Freq X")
        plt.ylabel("Freq Y")
        plt.colorbar(label='Magnitude')
        plt.tight_layout()
        plt.savefig(f"{sigma_folder}/hexfft_zoomed_sigma{sigma}_N{N}.png")
        plt.pause(T)
        plt.close()

        # Save point-based Gaussian map
        plt.figure(figsize=(8, 6))
        plt.imshow(map2, extent=extent2, origin='lower', cmap='hot')
        plt.colorbar(label='Gaussian Intensity')
        plt.title(f"Gaussian Map (Quasicrystal) ({txt_file}): σ={sigma}, N={N}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(f"{sigma_folder}/pointmap_sigma{sigma}_N{N}.png")
        plt.pause(T)
        plt.close()

        # Save point-based FFT (zoomed)
        fft2_res = fft2(map2)
        fft2_mag = np.abs(np.fft.fftshift(fft2_res))
        zoom_slice2 = dynamic_zoom_region(fft2_mag)
        plt.figure(figsize=(8, 6))
        plt.imshow(fft2_mag[zoom_slice2], cmap='viridis')
        plt.title(f"Zoomed FFT (Quasicrystal) ({txt_file}): σ={sigma}, N={N}")
        plt.xlabel("Freq X")
        plt.ylabel("Freq Y")
        plt.colorbar(label='Magnitude')
        plt.tight_layout()
        plt.savefig(f"{sigma_folder}/pointfft_zoomed_sigma{sigma}_N{N}.png")
        plt.pause(T)
        plt.close()
exit()

## Generates and plots the quasicrystal
if __name__ == '__main__':
    points, temp = gm.quasicrystal(35, 1)
    np.savetxt('points/points1.txt', points, fmt='%.6f', delimiter=',')
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-12, 12)

    ax.set_ylim(-12, 12)

    ax.plot(points[:,0],points[:,1],'.')
    ax.plot(temp[:,0],temp[:,1],'x', color = 'red')

    plt.show()
exit()



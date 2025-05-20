import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
import os

## Utilities used as a simplification of the other parts of the program

def deduplicate(points):
    """
    Deduplicates inserted points with a certain decimal accuracy.
    """
    rounded = np.round(points, 6)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(idx)]

def rotate_points(points):
    """
    Rotates points into position due to system symetry.
    """
    t = np.pi / 3
    rot_mat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    all_rotated = [points]
    for i in range(1, 6):  # 1 to 5
        rotated = points @ np.linalg.matrix_power(rot_mat, i).T
        all_rotated.append(rotated)
    return np.vstack(all_rotated)

def dynamic_zoom(arr, threshold_ratio=0.05, padding=10):
    """
    Dynamically zooms map to a square range with a peaks above 5% of the maximum peak
    """
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


def hexagonal(rows, cols, side):
    """
    Generates points for hexagonal grid
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

def plot_fft_and(map_data, extent, name_prefix, folder, sigma, N, radius, cor=False):
    """
    Plots maps and their zoomed fourier transform and saves the graphs to a specified directory
    """
    # Plot map
    plt.figure(figsize=(8, 6))
    plt.imshow(map_data, extent=extent, origin='lower', cmap='hot')
    plt.colorbar(label='Gaussian Intensity')
    if cor:
        plt.title(f"{name_prefix} Corrected Map (σ={sigma}, N={N}, R={radius})")
    else:
        plt.title(f"{name_prefix} Map (σ={sigma}, N={N}, R={radius})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    
    # Save map image
    map_filename = f"{name_prefix.lower().replace(' ', '_')}map_sigma{sigma}_N{N}"
    if cor:
        map_filename += "_corrected"
    map_filename += ".png"
    plt.savefig(os.path.join(folder, map_filename))
    plt.close()

    # Plot FFT
    fft = fft2(map_data)
    fft_mag = np.abs(np.fft.fftshift(fft))
    zoom_slice = dynamic_zoom(fft_mag)
    plt.figure(figsize=(8, 6))
    plt.imshow(fft_mag[zoom_slice], cmap='viridis')
    if cor:
        plt.title(f"{name_prefix} Corrected FFT Zoomed (σ={sigma}, N={N}, R={radius})")
    else:
        plt.title(f"{name_prefix} FFT Zoomed (σ={sigma}, N={N}, R={radius})")
    plt.xlabel("Freq X")
    plt.ylabel("Freq Y")
    plt.colorbar(label='Magnitude')
    plt.tight_layout()

    # Save FFT image
    fft_filename = f"{name_prefix.lower().replace(' ', '_')}fft_zoomed_sigma{sigma}_N{N}"
    if cor:
        fft_filename += "_corrected"
    fft_filename += ".png"
    plt.savefig(os.path.join(folder, fft_filename))
    plt.close()


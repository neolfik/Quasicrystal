# Fourier Transform of Quasicrystal Density Project

This is the README file for the Fourier Transform of Quasicrystal Density project.

---

## 📁 Folders

The folder contains two subfolders: `points` and `Saved_figures`.

### 🔹 Saved_figures

This folder contains plots of all generated grids, either quasicrystal or hexagonal.

### 🔹 points

This folder contains `.txt` files of generated points for the quasicrystal. This was needed due to the time required for quasicrystal generation with the first-generation programs. It can be rewritten to generate everything altogether, but this approach saves time for other users since the quasicrystal is already generated.

---

## 📄 Files

### 🔸 `add_gauss`

Used to add a density around the points that correspond with a Gaussian distribution.

### 🔸 `generate_mech_bcup2`

Used for the generation of the quasicrystal. It uses parallelization and polygon overlap checks, so the `concurrent.futures` and `shapely` packages are needed. Install them before generating the quasicrystal. If you do not want to generate the quasicrystal or do not want to download the packages, use the `.txt` files in the `points` subfolder.

### 🔸 `quasi_tiling`

Used to define the unit of the quasicrystal and its generation from three points, along with a few other point manipulation functions that are used more frequently.

### 🔸 `main`

This is the core program to be executed. It has two parts, and in its basic configuration, it is meant to only execute the interpretation of data and not generate the quasicrystal. For the generation, you need to bring the section of code after the first `exit()` to the beginning of the file and execute the Python program. The execution under `main` is used due to the nature of `concurrent.futures`.


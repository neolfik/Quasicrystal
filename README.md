
# Fourier Transform of Quasicrystal Density Project

This is the README file for the Fourier Transform of Quasicrystal Density project.

---

## 📁 Folders

The folder contains two subfolders: `points` and `Saved_figures`.

### 🔹 Saved_figures

This folder contains plots of all generated grids, either quasicrystal or hexagonal.

### 🔹 points

This folder contains `.txt` files of generated points for the quasicrystal. This was needed due to the time required for quasicrystal generation; the recursion of the first two to three repetitions is rather fast in a matter of a few minutes, thanks to parallelization.

---

## 📄 Files

### 🔸 `add_gauss`

Used to add a density around the points that correspond with a Gaussian distribution. It returns a map of specified accuracy. Be mindful when choosing the accuracy, as the computation time rises fast with the size of the map.

### 🔸 `recursive_generation`

Used for the generation of the quasicrystal. It uses parallelization with `concurrent.futures` to make the generation of higher repetitions. Install the package before generating the quasicrystal. If you do not want to generate the quasicrystal or do not want to download the packages, use the `.txt` files in the `points` subfolder. When generating the number of points rises very quickly at 4 repetitions, the program outputs around 1 million points, and at 5 repetitions, the program outputs approximately 12 million points, and the .txt file has 250 MB.

### 🔸 `utilities`

This file contains functions frequently used in other files. 

### 🔸 `main`

This is the core program to be executed. It has two parts, and in its basic configuration, it is meant to only execute the interpretation of data and not generate the quasicrystal. For the generation, you need to bring the section of code after the first `exit()` to the beginning of the file and execute the Python program. The execution under `main` is used due to the nature of `concurrent.futures`.


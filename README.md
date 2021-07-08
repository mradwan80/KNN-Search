# KNN-Search
This program implements a k-Nearest Neighbors search algorithm, based on using a structure named the DDS (discrete depth structure). In short, the DDS extends the cells of a view aligned 2D grid to depth piles. Points are mapped to the piles of the corresponding cells, and sorted by depth. A link to a journal paper describing the structure will be provided soon.

********************************************************************************

The program is cross platform (Windows and Linux). It was tested on:

-Windows 11 Home. build 22000.51.

-CMake GUI 3.21.0 (for windows build), with generator Visual Studio 14 2015

-WSL2 (for linux build test) Ubuntu-20.04

-Cuda compilation tools, release 9.1, V9.1.85

-GeForce GTX 950M



To run in linux, it is important to define these environment variables: CUDA_INCLUDE_DIRECTORY, CUDA_LIBRARY_DIRECTORY, CUDACXX.


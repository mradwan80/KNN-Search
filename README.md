# KNN-Search
This program implements a k-Nearest Neighbors search algorithm, based on using a structure named the DDS (discrete depth structure). In short, the DDS extends the cells of a view aligned 2D grid to depth piles. Points are mapped to the piles of the corresponding cells, and sorted by depth. A link to a journal paper describing the structure will be provided soon.


The program is cross platform (Windows and Linux).

********************************************************************************

## Running on Windows

Use CMake to generate a Visual Studio. Then open the ALL_BUILD project in Visual Studio IDE and build it. Then run the knn project.

The program was tested in a windows environment, with the following specifications:

* Windows 11 Home. build 22000.51.
* Cuda 10.2
* CMake GUI 3.21.0, with generator Visual Studio 14 2015
* GeForce GTX 950M


********************************************************************************

## Running on Linux

You will need to define the environment variables: 
* CUDA_INCLUDE_DIRECTORY, 
* CUDA_LIBRARY_DIRECTORY,
*  CUDACXX. 
  
Then move to the directory of the project, and:

* mkdir build
* cd build
* cmake ..
* make
* ./knn


The project was tested on WSL2 (Windows Subsystem for Linux 2), with the following specifications:

* Ubuntu-20.04
* Cuda 11.0
* gcc-9, g++-9
* CMake 3.16.3
* GeForce GTX 950M

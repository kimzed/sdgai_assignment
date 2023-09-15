## Introduction

This document explains the task assignment for a GIS fellow applicant for SDG AI Lab.

1. **Task 1** checks the candidate’s ability to work with simple geographical data visualization, editing, and analysis.
2. **Task 2** tests the candidates’ ability to work with raster data and perform commonly needed processes.
3. **Task 3** expects the candidate to work with pretrained models to perform scene classification and evaluation in Python.

The candidate is supposed to use Python scripts with any libraries desired. The delivered code is expected to be commented as much as possible. For any step delivered by any other method than Python (by hand with QGIS for example), only half points will be awarded. In this case, every step should be documented with screenshots and comments.

## Setting-up Development Environment

1. Create a private GitHub repository under your own account.
2. Please add the users `Skerre`, `yuctrn`, and `sdgailabtest` to your repository with read permission.
3. Upload your solution to the repository before the 17th of September 11:59 PM (GMT+3). Don't forget to add your `requirements.txt` or `.yaml` file for us to reproduce your environment.
4. Datasets are available [here](https://drive.google.com/drive/folders/1uJ2SfuFo4H561FPj97AHZwvwT6HjJ3rn?usp=sharing).

## Task 1 – Data Processing and Mapping (20 points)

The datasets for Task 1 are stored in the folder `TASK 1`.

1. Import the shapefile `points2.shp` to your project with all other necessary python libraries. (1 pt)
2. Create a new point at `-2.00144, 11.76553` with the following features attributes: URBAN_RURA = ‘U’; ALT_DEM = 42. (2 pts)
3. Remove obvious spatial outliers. (Hint: you can visualize the points and see the outliers). (2 pts)
4. Save the coordinates (LAT LON) only into a csv file. (2 pts)
5. Count the amount of Urban and Rural points in the extent: `[12.593677957; -2.001880227, -0.967547730, 12.118472459]`. (4 pts)
6. Plot the points within the extent on a map. Add a Basemap such as OSM. (2 pts)
7. Merge the point shapefile on the key `DHSCLUST` and `Cluster number` from the csv file `additional_data.csv` into one dataframe. Allow only unique values for cluster numbers. (2 pts)
8. Plot a scatterplot using `wealth index` and `number of household members`. What do you observe? (2 pts)
9. Plot two histograms in one cell using matplotlib. Left plot shows the column `number of household members` and the right plot `number of household members` standardized (z-score). (3 pts)

## Task 2 – Geographical Data Manipulation and Management (25 points)

Take the two available Sentinel 2 image folders from the data folder and perform data preprocessing tasks. We expect that the fellow creates a series of functions, preferably embedded within a reproducible virtual environment with a requirements file to be able to automatically deal with the core GIS operations:

1. Find and stack relevant raster bands found in the sentinel folder. (2.5pts)
2. Clip the image using the mask layer (boundary layer: `La Palma Bounds.geojson`). (5pts)
3. Rescale the bands (for example: 0-1 or 0-255). (5pts)
4. Resample the pixel size of bands 2,3,4,8 to 10 meter or 20 meter. (5pts)
5. Output a new `.tif` file based on the input bands. (5pts)
6. Plot the image. (2.5pts)

**Desired Outcome**: Stacked image with bands 2,3,4,8 + band 11 with a pixel resolution of 10m or 20m, clipped to the boundary of the `La_palma_bounds.geojson` file for the scene captured on 10.09.21 (D/M/Y). Band pixel values scaled to a range of (0-1 or 0-255).

## Task 3 – Scene Classification by Deep Learning (30 points)

The provided files in the Google Drive folder for this task are:

1. Weights file of the trained model.
2. A dataset folder consisting of images with 10 land use classes.

In this task, we'd like to ask the candidates to utilize the trained deep learning model to make predictions for the given images. Please follow the below steps:

1. Import necessary libraries, load the dataset, and visualize a sample image belonging to one of the land use classes in the Jupyter Notebook. (Hint: You can work with the Image module from the PIL for visuals and for modeling PyTorch). (2.5 pts)
2. Initiate a ResNet-50 model (pre-trained on the Imagenet dataset) and fine-tune your model with the provided model weights file. (Hint: You can use CPU as a device instead of GPU if your computer has no GPU). (5 pts)
3. Apply necessary transformations to the images before using your model to classify the scenes. Please explain why you made these transformations. (7.5 pts)
4. Pass the transformed image to the model, obtain the predicted classes, and visualize several images with corresponding classes. (7.5 pts)
5. Report accuracies/confusion matrices of your predictions (accuracy scores are not judged, but the interpretation and commentary by the candidate). (7.5 pts)


# SDG AI Lab GIS Assignment

GIS and deep learning assignment for the SDG AI Lab Fellow candidacy (September 2023).

## Overview

Solutions to a three-part technical assessment covering geospatial data processing, raster manipulation, and satellite image scene classification using a pretrained deep learning model. Built with a focus on code quality, reproducibility, and robust documentation.

## Tech Stack

- **Geospatial:** GeoPandas, Rasterio, Shapely
- **Deep Learning:** PyTorch (pretrained ResNet for scene classification)
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Code Quality:** Pylint, Black, Mypy, pytest
- **Environment:** Conda (Python 3.11)

## Tasks

1. **Data Processing and Mapping** — Visualization, editing, and statistical analysis of survey data (Burkina Faso wealth index vs. household size)
2. **Geographical Data Manipulation** — Raster data processing and transformation
3. **Scene Classification by Deep Learning** — Pretrained ResNet model for land-use classification achieving 97% accuracy across 10 classes

## Results

### Scene Classification (Task 3)

| Class               | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| AnnualCrop          | 0.91      | 0.96   | 0.93     |
| Forest              | 0.98      | 0.96   | 0.97     |
| HerbaceousVegetation| 0.96      | 0.96   | 0.96     |
| Highway             | 1.00      | 1.00   | 1.00     |
| Industrial          | 1.00      | 0.98   | 0.99     |
| Pasture             | 0.94      | 0.96   | 0.95     |
| PermanentCrop       | 0.94      | 0.90   | 0.92     |
| Residential         | 1.00      | 1.00   | 1.00     |
| River               | 0.98      | 0.98   | 0.98     |
| SeaLake             | 0.98      | 0.98   | 0.98     |
| **Overall Accuracy**| **0.97**  |        |          |

### Scatterplot Analysis (Task 1)

<img src="deliverables/task_1/scatter_plot_clustered.png" alt="Wealth Index vs. Household Members (clustered)" width="500" height="300"/>

Spatial clustering reveals a negative correlation (R² = 0.10) between wealth index and household size — significantly stronger than the unclustered data (R² = 0.01), highlighting the role of spatial aggregation in reducing noise.

## Project Structure

```
sdgai_assignment/
├── task_1.py              # Data processing and mapping
├── task_2.py              # Raster data manipulation
├── task_3.py              # Deep learning scene classification
├── deep_learning/         # Model and metrics
├── deliverables/          # Output plots, CSVs, rasters
├── test/                  # pytest test suite
├── utils/                 # Utility functions (raster, vector, DL, visualization)
└── environment.yml        # Conda environment
```

## Getting Started

```bash
conda env create -f environment.yml
conda activate your_env_name
python task_1.py
python task_2.py
python task_3.py
pytest .
```

Datasets available at this [Google Drive link](https://drive.google.com/drive/folders/1uJ2SfuFo4H561FPj97AHZwvwT6HjJ3rn?usp=sharing).

## Context

Technical assignment for the SDG AI Lab Fellow candidacy at UNDP. Demonstrates GIS analysis, raster processing, and deep learning capabilities with production-grade code quality standards (Pylint, Black, Mypy, pytest).

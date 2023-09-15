# SDG AI Lab GIS Assignment

Welcome to the GIS assignment repository for the SDG AI Lab Fellow Candidates of September 2023. The repository provides solutions to the tasks assigned, ensuring top-tier coding practices, optimal solutions, and robust documentation for easier evaluation. The codebase is structured in a manner that allows for simple reproduction and evaluation.

## Table of Contents

- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
- [Directory Structure](#directory-structure)
- [Tasks Overview](#tasks-overview)
- [Running the Scripts](#running-the-scripts)
- [Deliverables](#deliverables)
- [Acknowledgements](#acknowledgements)
- [Contact Information](#contact-information)

## Getting Started

### Prerequisites

- A Python environment (my environment ran on v3.11).
- Required packages and dependencies can be installed from the `environment.yml` file.
- Datasets are available at this [Google Drive link](https://drive.google.com/drive/folders/1uJ2SfuFo4H561FPj97AHZwvwT6HjJ3rn?usp=sharing).

### Setup

1. Clone the repository to your local machine.
2. Download the three data folders (e.g. 'TASK 2 Data-20230913T191955Z-001' for task 2) and move them into the local repo.
3. Navigate to the repository directory.
4. Set up a Python environment and activate it.
5. Install required packages: `conda env create -f environment.yml`
6. Activate the environment: `conda activate your_env_name`

## Directory Structure

A brief explanation of the repository's structure:

```
sdgai_assignment/
│
├── deep_learning/ # Contains deep learning model and metrics
│
├── deliverables/ - Contains all output files, plots, and results.
│
├── test/ - Test suites and cases ensuring the functions work correctly.
│
└── utils/ - Utility function files for raster, vector, deep learning, and visualization.
```


## Tasks Overview

1. **Data Processing and Mapping** - Consists of data visualization, editing, and simple analysis tasks.
2. **Geographical Data Manipulation and Management** - Focuses on raster data processing.
3. **Scene Classification by Deep Learning** - Utilizes a pretrained model to classify scenes.

Detailed task descriptions can be found in the 'GIS Recruitment Task Fall 2023.pdf' document from the repo

## Running the Scripts

- For Task 1: `python task_1.py`
- For Task 2: `python task_2.py`
- For Task 3: `python task_3.py`

Make sure to adjust to correctly configure PYTHONPATH to be in the repo.

## Deliverables

- All deliverable plots, CSVs, and raster images can be found in the `deliverables` directory.
- The assignment also outputs specific metrics and results directly to the console.

## Acknowledgements

I'd like to thank the SDG AI Lab for providing this challenging and enriching assignment. It offered an opportunity to showcase GIS and deep learning capabilities in a practical manner.

## Contact Information

- Candidate: BARON Cedric
- Email: cedric.baron.ls@gmail.com
- LinkedIn: [linkedin](https://www.linkedin.com/in/c%C3%A9dric-baron-ab846a156/)

For any further queries or feedback regarding this assignment, please feel free to reach out.

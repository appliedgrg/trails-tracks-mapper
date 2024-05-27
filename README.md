# Mapping Animal and Human Trails in Northern Peatlands using LiDAR and Convolutional Neural Networks

## Overview
This repository contains the code and data for the paper titled "Mapping Animal and Human Trails in Northern Peatlands using LiDAR and Convolutional Neural Networks." The goal of this project is to develop and validate tools for automatically mapping trails from high-resolution remote-sensing imagery using Convolutional Neural Networks (CNNs) and LiDAR data.

### Background
Trails, which are detectable signs of passage left by individuals or groups, play a crucial role in understanding the presence and movement of organisms in various landscapes. These trails can be transient or semi-permanent, depending on the substrate and frequency of use. While most contemporary researchers track wildlife using methods such as GNSS, camera traps, or genetics, many studies could benefit from additional spatial information on the explicit location of trails.

### Objective
The primary objective of this research is to develop a fully automated strategy for detecting and mapping trails and related linear features using modern datasets and processing algorithms. The study compares optical and LiDAR data to evaluate the effectiveness of CNNs in mapping trails in northern peatlands.

### Methodology
- **Data Acquisition**: The study area is located in the boreal zone of northeastern Alberta, Canada. High-resolution LiDAR and RGB imagery were collected using drone and aerial platforms.
- **Model Architecture**: A U-Net model, a type of CNN renowned for its efficacy in image segmentation tasks, was employed. The model was configured to accept one-band input images of 256x256 pixels, with batch normalization and a dropout rate of 0.3 to prevent overfitting.
- **Training and Validation**: The model was trained on manually labeled training data, with various data augmentation techniques applied to increase dataset variability. The performance of the model was evaluated using visual interpretation of high-resolution imagery and field inspections.

### Results
The U-Net model demonstrated high accuracy in mapping trails across different terrains, achieving an overall accuracy of 91%, with a precision of 79% and a recall of 83%. The study found that LiDAR data, due to its ability to penetrate ground vegetation, was superior to high-resolution RGB imagery in detecting trails, particularly under dense canopies.

### Applications
The developed tools and models can significantly enhance ecological monitoring and conservation efforts by providing detailed spatial data on animal and human trails. The trail maps generated can be used to better understand animal behavior, design more effective wildlife monitoring studies, and assess the impact of human activities on natural landscapes.

For more detailed information on the methodology, data, and results, please refer to the paper and supplementary materials included in this repository.

## Directory Structure
The repository is organized as follows:

trails-tracks-mapper
│ README.md
│ LICENSE
│ .gitignore
│
├───examples
│ ├── raw_patches
│ ├── aerial_raster
│ └── CNN_predictions
│
├───notebooks
│ └── preprocessing_training_prediction.ipynb
│
├───models
│ ├── Trails_Tracks_DTM10cm_512px.h5
│ ├── Trails_Tracks_DTM50cm_256px.h5
│ └── Trails_Tracks_DTMnorm50cm_512px.h5
│
├───scripts
│ ├── data_preprocessing.py
│ ├── data_generator.py
│ ├── train.py
│ └── predictions.py
│
└───docs
├── figures
└── supplementary_material


## Installation
To set up the environment and install dependencies, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/appliedgrg/trails-tracks-mapper.git
   cd trails-tracks-mapper

2. **Create a virtual environment and install dependencies**:
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

## Best Performing CNN Models

1. **DTM 10 cm with 512 px window size**:

Trails_Tracks_DTM10cm_512px

2. **DTM 50 cm with 512 px window size**:

Trails_Tracks_DTM50cm_256px

3. **DTM normalized 50 cm with 512 px window size**:

Trails_Tracks_DTMnorm50cm_512px


## Contributors
Irina Terenteva (irina.terenteva@ucalgary.ca)
Xue Yan Chan (mcdermid@ucalgary.ca)
Gregory J. McDermid (mcdermid@ucalgary.ca)

## License
This project is licensed under the MIT License - see the LICENSE file for details.


# Mapping Animal and Human Trails in Northern Peatlands using LiDAR and Convolutional Neural Networks

## Overview
This repository contains the code and data for the paper titled "Mapping Animal and Human Trails in Northern Peatlands using LiDAR and Convolutional Neural Networks." The goal of this project is to develop and validate tools for automatically mapping trails using Convolutional Neural Networks (CNNs) and LiDAR data (both drone and aerial platforms).

### Background
Trails are physical signs of movement left by animals or humans and are important for studying wildlife behavior and human impact on natural environments. These trails can vary in permanence, influenced by factors such as terrain and frequency of use. Traditional wildlife tracking methods include GNSS, camera traps, and genetic analysis, but these methods can be enhanced with detailed spatial data on trail locations.

Understanding the exact locations of trails can provide valuable insights for ecological research, such as habitat selection and movement patterns, and can also aid in conservation efforts. Despite their importance, there is limited research on automatically detecting and mapping trails using remote sensing data. This project leverages advanced techniques, specifically CNNs and LiDAR data, to fill this gap and provide accurate trail maps in northern peatlands.

### Objective
The primary objective of this research is to develop a fully automated strategy for detecting and mapping trails and related linear features using modern datasets and processing algorithms. The primary objectives of this research are:
>> To demonstrate the capacity of high-density LiDAR and CNNs to map trails and tracks automatically.
>> To compare the accuracy of trail and track maps developed with LiDAR data from drone and piloted-aircraft platforms.
>> To measure the abundance and distribution of trails and tracks across different land-cover classes and their co-location with anthropogenic disturbances in the boreal forest of northeastern Alberta, Canada.

### Methodology
- **Data Acquisition**: The study area is located in the boreal zone of northeastern Alberta, Canada. High-resolution LiDAR and RGB imagery were collected using drone and aerial platforms.
- **Input Data**: To train the CNN model, high-resolution digital terrain models (DTMs) derived from LiDAR data were used. The DTMs capture fine-scale variations in the terrain surface, making them ideal for identifying linear features such as trails and tracks.
- **Model Architecture**: A U-Net model, a type of CNN renowned for its efficacy in image segmentation tasks, was employed. The model was configured to accept one-band input images of 256x256 and 512x512 pixels, with batch normalization and a dropout rate of 0.3 to prevent overfitting.
- **Training and Validation**: The model was trained on manually labeled training data, with various data augmentation techniques applied to increase dataset variability. The performance of the model was evaluated using visual interpretation of high-resolution imagery and field inspections.

### Results
The study demonstrated that high-density LiDAR and CNNs could accurately map trails and tracks across a diverse boreal forest area. Maps developed using LiDAR data from both drone and piloted-aircraft platforms showed no significant difference in accuracy. The piloted-aircraft LiDAR map achieved an F1 score of 77% ± 9%.

The research identified a 2829-km network of trails and tracks within the 59-km² study area, with a higher concentration in peatlands. The study also revealed that seismic lines significantly influence movement patterns.

[![CNN Model Applied to UAV Data](https://github.com/appliedgrg/trails-tracks-mapper/blob/main/images/CNN_at_UAVdata.png)](https://github.com/appliedgrg/trails-tracks-mapper/blob/main/images/CNN_at_UAVdata.png)


### Applications
The developed tools and models can significantly enhance ecological monitoring and conservation efforts by providing detailed spatial data on animal and human trails. The trail maps generated can be used to better understand animal behavior, design more effective wildlife monitoring studies, and assess the impact of human activities on natural landscapes.

For more detailed information on the methodology, data, and results, please refer to the paper which was submitted to PeerJ.


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
This project is licensed under the Creative Commons BY-NC 4.0 License - see the LICENSE file for details.


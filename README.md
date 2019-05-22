# CURE-OR

The goal of this project is to analyze the robustness of off-the-shelf recognition applications under multifarious challenging conditions, investigate the relationship between the recognition performance and image quality, and estimate the performance based on hand-crafted features as well as data-driven features. To achieve this goal, we introduced a large-scale, controlled, and multi-platform object recognition dataset CURE-OR, which includes 1 million images of 100 objects captured with different backgrounds, devices and perspectives, as well as simulated challenging conditions. This repository includes codes to produce analysis results in our papers. For more information about CURE-OR, please refer to our papers and website linked below.

* Papers
  * [CURE-OR: Challenging Unreal and Real Environments for Object Recognition](https://arxiv.org/pdf/1810.08293.pdf)
  * [”Object Recognition Under Multifarious Conditions: A Reliability Analysis and A Feature Similarity-Based Performance Estimation](https://arxiv.org/pdf/1902.06585.pdf)
* Website
  * [OLIVES](https://ghassanalregib.com/)
  * [CURE-OR dataset page](https://ghassanalregib.com/cure-or/)

## Dataset
Objects of CURE-OR: 100 objects of 6 categories
<p align="center"><img src="./figs/cureor_objects.png", width="800"></p>

5 Backgrounds: White, 2D Living room, 2D Kitchen, 3D Living room, 3D Office
<p align="center"><img src="./figs/cureor_backgrounds.png", width="600"></p>

5 Devices: iPhone 6s, HTC One X, LG Leon, Logitech C920 HD Pro Webcam, Nikon D80
<p align="center"><img src="./figs/cureor_devices.png", width="600"></p>

5 Object orientations: Front, Left, Back, Right, Top
<p align="center"><img src="./figs/cureor_object_orientations.png", width="600"></p>

In order to receive the download link, please fill out this [form](https://goo.gl/forms/YVM3N6RrywNPuEjJ3) to submit your information and agree to the conditions to use. These information will be kept confidential and will not be released to anyone outside the OLIVES administration team.

## Usage
Download the analysis data from [here](https://www.dropbox.com/s/76ilscx5zha3imf/cure_or_analysis_data.tar.gz?dl=0) and unzip it under the same directory as the codes. The folder structure is as following:
```
├── AWS/                          # Recognition results from AWS Rekognition API
│    ├─── 01_no_challenge/        # Organized in folders by challenge types of CURE-OR
│    └─── ...
├── Azure                         # Recognition results from Microsoft Azure Computer Vision API
│    ├─── 01_no_challenge/        # Organized in folders by challenge types of CURE-OR
│    └─── ...
├── IQA                           
│    ├── IQA_codes                # Matlab codes for image quality assessments
│    └── Result                   # Image quality results organized in folders by objects
└── CBIR                          # Content-based image retrieval
     ├── Features                 # Extracted features
     ├── Performance              # Performance of recognition applications preprocessed for analysis
     └── Distance                 # Distance between features of "best" images and the rest: averaged across objects
```
CBIR codes were referenced from [this repo](https://github.com/pochih/CBIR).


To see the analyis results, simply run:
```
python analysis.py
```
The results will be stored under ```Results/```.

## Citations
If you use CURE-OR dataset and/or these codes, please consider citing our papers
```
@inproceedings{Temel2018_ICMLA,
author      = {D. Temel and J. Lee and G. AlRegib},
booktitle   = {2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA)},
title       = {CURE-OR: Challenging unreal and real environments for object recognition},
year        = {2018},}

@INPROCEEDINGS{Temel2019_ICIP,
author      = {D. Temel and J. Lee and G. AIRegib},
booktitle   = {IEEE International Conference on Image Processing (ICIP)},
title       = {Object Recognition Under Multifarious Conditions: A Reliability Analysis and A Feature Similarity-Based Performance Estimation},
year        = {2019},}
```

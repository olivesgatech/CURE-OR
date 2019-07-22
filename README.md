# CURE-OR

<p align="center"><img src="./figs/cure_or_lr.gif", width="500"></p>


The goal of this project is to analyze the robustness of off-the-shelf recognition applications under multifarious challenging conditions, investigate the relationship between the recognition performance and image quality, and estimate the performance based on hand-crafted features as well as data-driven features. To achieve this goal, we introduced a large-scale, controlled, and multi-platform object recognition dataset CURE-OR, which stands for Challenging Unreal and Real Environments for Object Recognition. In CURE-OR dataset, there are 1,000,000 images of 100 objects with varying size, color, and texture, captured with multiple devices in different setups. The majority of images in the CURE-OR dataset were acquired with smartphones and tested with off-the-shelf applications to benchmark the recognition performance of devices and applications that are used in our daily lives. This repository summrizes the characterisitcs of our dataset and provides codes to reproduce analysis results in our papers. For more information about CURE-OR, please refer to our papers.

* Papers
  * [CURE-OR: Challenging Unreal and Real Environments for Object Recognition](https://arxiv.org/pdf/1810.08293.pdf)
  * [Object Recognition Under Multifarious Conditions: A Reliability Analysis and A Feature Similarity-Based Performance Estimation](https://arxiv.org/pdf/1902.06585.pdf)
* Website
  * [OLIVES](https://ghassanalregib.com/)

## Dataset
<table>
<tbody>
<tr style="text-align:justify;">
<td style="text-align:center;"><strong> Object classes
(number of objects/class)</strong></td>
<td style="text-align:center;"><strong>Images
per object</strong></td>
<td style="text-align:center;"><strong>Controlled condition
(level)</strong></td>
<td style="text-align:center;"><strong>Backgrounds</strong></td>
<td style="text-align:center;"><b>Acquisition devices</b></td>
<td style="text-align:center;"><b>Object orientations</b></td>
</tr>
<tr>
<td style="text-align:center;">Toy (23)
Personal (10)
Office (14)
Household (27)
Sports/Entertainment (10)
Health (16)</td>
<td style="text-align:center;">10,000</td>
<td style="text-align:center;">Background (5)
Object orientation (5)
Devices (5)
Challenging conditions (78)</td>
<td style="text-align:center;">White 2D (1)
Textured 2D (2)
<span style="font-family:inherit;font-size:inherit;">Real 3D (2)</span></td>
<td style="text-align:center;">DSLR: Nikon D80
Webcam: Logitech C920
Smartphones: iPhone 6s, HTC One, LG Leon</td>
<td style="text-align:center;">Front (0<sup>o</sup>)
Left side (90<sup>o</sup>)
Back (180<sup>o</sup>)
Right side (270<sup>o</sup>)
Top</td>
</tr>
</tbody>
</table>





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

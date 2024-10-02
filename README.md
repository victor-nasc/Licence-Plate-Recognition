# Vehicle License Plate Recognition in Videos

This repository is the official implementation of **Efficient License Plate Recognition in Videos Using Visual Rhythm and Accumulative Line Analysis** paper

**Authors**: Victor Nascimento Ribeiro and Nina S. T. Hirata

![Alt Text](https://i.imgur.com/NhFnMvW.png)

The work participated in 
- [**CVPR 2024**](https://cvpr.thecvf.com/) - [LatinX in CV Undergraduate Consortium](https://www.latinxinai.org/cvpr-2024): Work in Progress paper [PDF](https://drive.google.com/file/d/1ERIpwWajpBXPXiLKC9gUW9ulL1JeXA82/view?usp=sharing)
- [**SIBIGRAPI 2024**](https://sibgrapi.sbc.org.br/2024/): Workshop of Undergraduate Works full paper [PDF](https://drive.google.com/file/d/1u7Ts3SMMgrtBTOAFP6qwAIi9_IT2X-_q/view?usp=sharing)



<br>


## Description

**This work presents an alternative and more efficient method for recognizing vehicle license plates in videos using Deep Learning and Computer Vision techniques.**

> Conducted at the [University of São Paulo - USP](https://www5.usp.br/) under the guidance of [Prof. Nina S. T. Hirata](https://www.ime.usp.br/nina/).

Video-based Automatic License Plate Recognition (ALPR) involves extracting vehicle license plate information from video captures. Traditional systems often require substantial computing resources and analyze multiple frames to identify license plates, resulting in high computational overhead. In this study, we propose two methods designed to efficiently extract license plate information from exactly one frame per vehicle, thereby significantly reducing computational demands.

The first method utilizes Visual Rhythm (VR) to generate time-spatial images from videos. The second method employs Accumulative Line Analysis (ALA), a novel algorithm that processes single-line video frames for real-time operations. Both methods utilize YOLO for license plate detection within the frame and Convolutional Neural Networks (CNNs) for Optical Character Recognition (OCR) to extract textual information.

Experimental results on real-world videos demonstrate that our methods achieve comparable accuracy to traditional frame-by-frame approaches while processing at speeds three times faster.

![Alt Text](https://i.imgur.com/7JVoYKI.png)


<br>




## Usage

This codebase is written for ```python3.9```

```bash
# Clone this repository
git clone https://github.com/victor-nasc/Vehicle-Licence-Plate-Recognition.git

# Install dependencies
pip install -r requirements.txt


# Run Visual Rhythm (VR) approach
python3 vr-alpr.py --OPTIONS

# --OPTIONS
#    --line_height: line height position             [defalt:  1000]
#    --interval: interval between VR images (frames) [default: 900]
#


# Run Accumulative Line Analysis (ALA) approach
python3 ala-alpr.py --OPTIONS

# --OPTIONS
#   --line_height: line height position                      [defalt:  1000]
#   --gamma: threshold to discards clusters and remove noise [default: 100]
#   --hide: hide the video from being shown realtime         [default: True]

#    The video path is prompted during execution.
```



<br>



## Citation

If you find the code useful in your research, please consider citing our paper:

```
...

```

**Contact**: victor_nascimento@usp.br 



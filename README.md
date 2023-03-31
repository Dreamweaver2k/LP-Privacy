# License Plate Privacy

## Overview

### Introduction

This project is derived from my Junior Independent work at Princeton University as a part of the Fairness in Computer Vision Seminar. This project seeks to provide a license plate privacy architecture that could be used in future work. For license plate privacy in video, it is important to have a real-time solution so that frames can be processed as they are captured. Additionally, it is important to make sure there are no false negatives since even one false negative can cause a privacy breach.

### Motivation

The motivation of this project comes from a variety of places. One of the key motivating factors is the reported abuses by authorities and individuals using license plates captured on security footage to blackmail individuals who are engaging in activities or interacting with people they do not want family or friends to know about. One example of this is a corrupt police officer checking the security footage in the parking lot of a gay bar and using license plates to identify individuals and find those married to women[^3].

### Components

The general components of this architecture are as follows:

- Detection
- Tracking
- Obfuscation

#### Detection

For detection, I trained a neural network on a collection of data available in the training_data folder. I selected a small YOLOv5 model architecture due to its incredible real-time performance and short training times.

#### Tracking

For license plate tracking, I implemented a Scale Invariant Feature Transform (SIFT) feature tracking algorithm. For each detected license plate from the previous component, the SIFT features are extracted and the approximate velocity of the object is calculated using optical flow. If a license plate goes from detected in one frame to undetected in another frame, then the velocity is envoked to search for the SIFT features in a region of interest. This improves recall of the system and reduces false negatives.

#### Obfuscation

To obfuscate the license plate, a simple pixel shuffling method is utilized. For each plate, a binary key is generated. Each bit represents a chunk of the image, and chunks corresponding to adjacent 1's are swapped to obfuscate the image.

## Installation

### Dependencies

Tesseract-ocr and libtesseract-dev are required some of the functionalities to work properly. You can download these in the following ways.

##### macOS

```bash
brew install tesseract
brew install --HEAD tesseract
```

##### Ubuntu

```bash
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

##### Windows

I suggest viewing https://tesseract-ocr.github.io/tessdoc/Downloads.html

##### Deactivate Tesseract

Alternatively, you can deactivate ocr functionality using

### Download

To download the repository, run the following commands in the directory of your choice:

```bash
git clone https://github.com/Dreamweaver2k/LP_privacy.git
```

You can now download the python dependencies by running:

```bash
cd ./LP_privacy
pip install -r requirements.txt
```

## Usage

### Running Locally

To run the detection and obfuscation model on a video, navigate to the LP_privacy directory, ensure all the requirements are installed, and run:

```bash
python detect.py --source destination/to/video/folder --weights weights/best.pt
```

This will run the detection model at .2 confidence threshold. If you want to change this threshold, run it with the conf-thresh parameter.

```bash
python detect.py --source destination/to/video/folder --weights weights/best.pt --conf-thres .5
```

Note that the source flag must point to a folder containing your video file. You can run it on our provided video file using:

```bash
python detect.py --source video/original_lp_footage.mp4 --weights weights/best.pt
```

### Running on Google Colab

The license plate privacy detection and blurring system can be tested for personal use on Google Colab. First, ensure that the proper packages are installed in the Colab runtime:

```python
!pip install opencv-contrib-python==4.4.0.44
!apt install tesseract-ocr
!apt install libtesseract-dev
!pip install pytesseract
!git clone https://github.com/Dreamweaver2k/LP_privacy.git
```

Once these have finished loading, you will have all the necessary files to detect and blur license plates in your personal videos as well as the the dependent packages. Now, you can run detect.py to perform the privacy function on your specified video:

```python
!python LP_privacy/LP_privacy/detect.py --source destination/to/video/folder --weights LP_privacy/LP_privacy/weights/best.pt
```

### Training New Weights

To train new weights, you will need to download YOLOv5 [^2]. Once this is done, you can utilize the Chinese City Parking Dataset [^1] (CCPD) which I have annotated for use in a YOLO model. See reference for original dataset.
To download YOLOv5, run the following in Google Colab:

```python
!git clone https://github.com/ultralytics/yolov5
```

To train new weights in Google Colab, run the following:

```python
%load_ext tensorboard
%tensorboard --logdir runs/

!python yolov5/train.py --img 416 --batch 4 --epochs 20 --data LP_privacy/LP_privacy/training_data/data.yaml --cfg yolov5/models/yolov5l.yaml --name lpmodel
```

Note, for the cfg flag, the users can select whatever YOLO model size they want. Smaller architectures train and deploy faster. Larger architectures may provide better performance.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[^1]:
    Z. Xu et al., “Towards end-to-end license plate detection and recognition: A large dataset and baseline,” in
    Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 255–271.

[^2]: [Online]. Available: https://pytorch.org/hub/ultralytics_yolov5/
[^3]: [Online]. Available: https://www.aclu.org/news/privacy-technology/documents-uncover-nypds-vast-license-plate-reader-database#:~:text=A%20police%20officer%20in%20Washington,were%20at%20a%20gun%20show.

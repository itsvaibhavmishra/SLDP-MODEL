# SLDP (SIGN LANGUAGE DETECTION PROGRAM)

A python program for Real-Time Hand Sign Detections made using Tensorflow API.

## Installation :

Move inside ```bash RealTimeObjectDetection\ ``` folder and run the following command on any terminal

```bash
python start.py
```
Wait for the program to completely initialize and a window will pop up

To exit out of program press 'q' key once.

Note: if webcam doesn't work then run this code to check your device number for webcam :

## CODE :

```python
import cv2 as cv 
def testDevice(source):
    cap = cv.VideoCapture(source) 
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
testDevice(0) # no printout
testDevice(1) # prints message 
```
and then change VideoCapture(0) code with your deive cam number

## Requirements :
```bash
OpenCV

Numpy

TensorFlow

LabelImg

Visual Studio

Protoc

TensorFlow Models
```
### Optional :
```bash
Anaconda

Jupyter notebook

CUDA

CUDNN
```
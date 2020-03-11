# Lane Detection

This is the python implementation of lane detection used in autonomous vehicles. The techniques used in this work are geometric vision based and are not robust for all scenarios.

## Dependencies
This code was tested on a system with following packages:

- Python - 3.6
- OpenCV - 3.4.5
- Numpy
- Matplotlib

## Directory Tree
The directory list for the dataset is given below

```
.
├── code2.py
├── code.py
├── Data1.mp4
├── Data2.mp4
├── Dataset
    ├── data_1
    │   ├── camera_params.yaml
    │   └── data
    │       ├── 0000000000.png
    │       ├── 0000000001.png
    └── data_2
        ├── cam_params.yaml
        └── challenge_video.mp4

```
## Instructions to run the code

- The python script `code.py` is the lane detection script for the dataset 1. To run the code, open a new terminal in the directory and type
```
python3 code.py
```
- The python script `code1.py` is the lane detection script for the dataset 2. To run the code, open a new terminal in the directory and type
```
python3 code2.py
```

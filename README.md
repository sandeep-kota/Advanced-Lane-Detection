# Lane Detection for Autonomous Vehciles

This is the python implementation of lane detection used in autonomous vehicles. The techniques used in this work are geometric vision based and are not robust for all scenarios.

 ![alt text](./Results/Lane_Detection_Demo.gif?raw=true "Demo 1")(https://www.youtube.com/watch?v=E52yUwqryhY&ab_channel=SandeepKota)

 ![alt text](./Results/Lane_Detection_Demo_2.gif?raw=true "Demo 2")(https://www.youtube.com/watch?v=CTZXtOfOYFI&ab_channel=SandeepKota)

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

## Working 

 ![alt text](./Results/ENPM673_Project2-1.jpg?raw=true "Working")


## Instructions to run the code for Improving Video Quality
- The python script `improve_video_quality_ques1.py` runs on every individual frame of the video.
```
python3 improve_video_quality_ques1.py ../Data/Night\ Drive\ -\ 2689_Problem1.mp4
```

## Instructions to run the code for Lane Detection

- The python script `code.py` is the lane detection script for the dataset 1. To run the code, open a new terminal in the directory and type
```
python3 code.py
```
- The python script `code1.py` is the lane detection script for the dataset 2. To run the code, open a new terminal in the directory and type
```
python3 code2.py
```

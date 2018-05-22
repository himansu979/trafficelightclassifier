# Traffic Light Classifier

### About this
This tutorial is about creating tensorflow object detection model for traffic light classifier. This will not only identify traffic lights in the image, but also it's state (ex: green, yellow, red).

We will follow the source from medium post : [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

Useful links for this project: <br>

* [github repository for tensorflow models](https://github.com/tensorflow/models)
* [Installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
* [Installing TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)
* [TensorFlow official website](https://www.tensorflow.org/install/install_windows)

Check if tensorflow is installed or not and its current version
```
python
>>> import tensorflow as tf
>>> tf.__version__
'1.7.0'
```

Use the following commands to clone the tensorflow/models repository into your local computer. This will clone the folder **models** into your current folder. These instructions are specific to windows system.
```
git clone https://github.com/tensorflow/models.git 
```
Navigate into the **models/research** folder and add the current directory and **slim** directory into `PYTHONPATH`
```
cd models\research
set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\slim
echo %PYTHONPATH%
%PYTHONPATH%;C:\Users\...\models\research;C:\Users\...\models\research\slim
```
To make sure that PATH is correctly set, type the following in the terminal inside `python` shell. The output should show the above two paths in the list.
```
python
>>> import sys
>>> sys.path
```

### Selecting a pre-complied model

Useful links for pre-complied/pre-trained models

* [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
* [COCO : Common Objects in Context](http://cocodataset.org/#home)
* [Object classes in MSCOCO dataset](https://stackoverflow.com/questions/42480371/i-want-to-know-if-there-is-the-clothing-object-class-in-the-ms-coco-dataset)
* [Arxiv: Microsoft COCO: Common Objects in Context paper, arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
* [PASCAL VOC (Visual Object Classes) page](http://host.robots.ox.ac.uk/pascal/VOC/)

We will use **ssd_mobilenet_v1_coco**, which is simplest and fastest pre-trained model. This is trained with COCO dataset with **80 object categories**.
Download this model from the following link and extract inside *models\research\object_detection* directory.
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz

Navigate to the extracted directory and locate these files.
```
cd ssd_mobilenet_v1_coco_2017_11_17/
ls
 checkpoint
 frozen_inference_graph.pb
 model.ckpt.data-00000-of-00001
 model.ckpt.index
 model.ckpt.meta
 saved_model/
   saved_model.pb
   variables/
```

Installation Instruction for Step by Step TensorFlow Object Detection API Tutorial — Part 3: Creating Your Own Dataset

Windows + Anaconda
Download and install Anaconda (Python 3+)

Open the Anaconda Prompt and go to the labelImg directory

1. conda install pyqt=5
2. pyrcc5 -o resources.py resources.qrc
3. python labelImg.py
4. python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

4th step is not neccessary.



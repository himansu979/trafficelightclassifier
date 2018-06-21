# Traffic Light Classifier

### About this
This tutorial is about creating tensorflow object detection model for traffic light classifier. This will not only identify traffic lights in the image, but also it's state (ex: green, yellow, red).

We will follow the source from medium post : [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

Useful links for this project: <br>

* [github repository for tensorflow models](https://github.com/tensorflow/models)
* [Installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
(inside models\research\object_detection\g3doc folder)
* [Installing TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)
* [TensorFlow official website](https://www.tensorflow.org/install/install_windows)

Check if tensorflow is installed or not and its installed version (current version is **1.8.0** as of 05/22/2018.
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

Navigate to the extracted directory and locate these files. Each pre-trained model contains a frozen inferefence graph (.pb file) and three checkpoint (.ckpt) files.
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

### Running the Object detection Demo

Run the following jupyter notebook to classify objects in the image. You have to run this inside `research/object_detection` directory.

https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

You can modify the parameter **MODEL_NAME** to try different models. This will use the images inside `object_detection\test_images`. Try experimenting with new images and extend the upper limit of `range(1, *)` (code block under Detection) to include new images.
```
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
```

### Part 2 : Creating TFRecord File from .yaml file

The .yaml file contains *path* to the images and *boxes* for each object in the image. Each box contains *label* and bounding box co-ordinates (*x_max, x_min, y_max, y_min*) for each object.

- Creating you own dataset : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
- https://github.com/swirlingsand/deeper-traffic-lights/blob/master/data_conversion_bosch.py

modify `LABEL_DICT` to include the objects in your images, `INPUT_YAML=path to the .yaml file`. Make sure the `path:` inside the .yaml file points to the images.

```
python tf_record.py --output_path training.record
```
*Note :

Step 1) Create YAML file using : https://gist.github.com/WuStangDan/e2484a7d27fb5d9f2d9201a4adcf99ea#file-bosch-train-yaml
Make sure space and indentation are correct

Step 2) Script can be created by referring :
https://github.com/swirlingsand/deeper-traffic-lights/blob/master/data_conversion_bosch.py

### Part 3 : TensorFlow Object Detection API Tutorial - Creating Your Own Dataset

LabelImg is a graphical image annotation tool and label object bounding boxes in image

https://github.com/tzutalin/labelImg


Windows + Anaconda
Download and install Anaconda (Python 3+)

Open the Anaconda Prompt and go to the labelImg directory

1. conda install pyqt=5
2. pyrcc5 -o resources.py resources.qrc
3. python labelImg.py
4. python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

4th step is not neccessary.

Ubuntu + Anaconda(Python 3 ) :

1. pip install PyQt5

2. pyrcc5 -o resources.py resources.qrc

3. pip install lxml

4. python labelImg.py

This will generate an **.xml** file for each image containing annotations for all objects inside the image. TensorFlow provides a script to (**create_pascal_tf_record.py**) convert this xml to TFRecords format. 

https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

### Part 4 : Training the Model
For additional information, follow this blog post : http://androidkt.com/train-object-detection/






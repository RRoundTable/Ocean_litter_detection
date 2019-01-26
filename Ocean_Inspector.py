from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image # Image : open image file
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import base64 # image encoding
import io
import PIL.Image, PIL.ImageTk
import threading
from Inspector_module import Ocean_Inspector # module 심기


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using

#'faster_rcnn_resnet101_coco_2018_01_28'
#'MODEL_NAME = faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_NAME = 'inference_graph_1'
VIDEO_NAME = 'video/ocean_litter.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

print(PATH_TO_CKPT)
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_1/','labelmap.pbtxt')

print(PATH_TO_LABELS)
# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

print(PATH_TO_VIDEO)
# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#imgLabel=Label(window, width=400, height=400, text="image print",bg="red") # image를 보여줄 장소

# label_map,categories,category_index,sess,imgLabel
Inspector=Ocean_Inspector(label_map,categories,category_index,sess,image_tensor,detection_boxes,detection_scores,detection_classes,num_detections)


#threading.Timer(1,Inspector.runModel).start()
# Inspector.window.after(20,Inspector.runModel)
Inspector.window.mainloop() # 마지막 완성
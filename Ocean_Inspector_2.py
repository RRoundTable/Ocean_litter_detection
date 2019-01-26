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

window=Tk() # window 설정
window.title("Image Viewer") # title
#window.geometry("640x480") # size
window.resizable(False,False)  # sizable



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

# Load the label map : litter ,fish, human
# Label maps map indices to category names, so that when our convolution
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


# file select
def fileSelect():
    global video
    selFile = filedialog.askopenfilename(initialdir="/", title='Select file',
                                         filetypes=(("Cideo files", "*.mp4;*.flv;*.avi;*.mkv"), ("all files", "*.*"))) # 동영상 로드
    video = cv2.VideoCapture(selFile) 
    return video


def stream(imgLabel):
    #global video
    score_list = []
    while (video.isOpened()):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # reulst

        score_list.append((boxes, scores, classes, num))
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)

        # frame 크기 조정 : computing power issue
        # if frame.shape[1]>frame.shape[0]:
        #     Vsize=400
        #     Hsize=int(frame.shape[0]/frame.shape[1]*400)
        # else:
        #     Hsize=400
        #     Vsize = int(frame.shape[1] / frame.shape[0] * 400)
        # frame=cv2.resize(frame,(Hsize,Vsize),interpolation = cv2.INTER_AREA)
        # tk photo image로 바꿔야 하는지 살펴보기
        #print("frame shape : " ,frame.shape) # 확인
        #print(frame)
        #frame=frame.tobytes()

        frame=PIL.ImageTk.PhotoImage(PIL.Image.fromarray(frame)) 
        canvas.create_image(0,0,image=frame, anchor=NW)
        #print("frame : ",frame)

        canvas.pack()
        imgLabel.image=frame
        imgLabel.configure(image=frame)  
        # cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    score_df=pd.DataFrame(score_list)



imgLabel=Label(window, width=400, height=400, text="image print",bg="red") # image를 보여줄 장소

resultLable=Label(window, width=400, height=400, bg='blue',text="result") # result Label : 오염도를 보여줄 Label

canvas=Canvas(window,width=700,height=400, bg='#afeeee')
canvas.pack()
imgLabel.pack()
resultLable.pack()
# 메뉴 구성하기
menubar=Menu(window) # window에 메뉴바 새성
menu_1=Menu(menubar, tearoff=0)
menu_1.bind("<<MenuSelect>>")

menu_1.add_command(label="Open",command=fileSelect) # fileSelect function
menubar.add_cascade(label="File",menu=menu_1)  # 클릭하면 "File" 뜬다
window.config(menu=menubar)
thread = threading.Thread(target=stream, args=(imgLabel,)) # function , label
thread.daemon = 1
thread.start()
window.mainloop() # 마지막 완성

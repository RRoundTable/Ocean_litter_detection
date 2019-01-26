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
from utils import label_map_util
from utils import visualization_utils as vis_util
import time


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


class Ocean_Inspector: # GUI Class

    def __init__(self,label_map,categories,category_index,sess,image_tensor,detection_boxes,detection_scores,detection_classes,num_detections):
        self.label_map=label_map
        self.categories=categories
        self.category_index=category_index
        self.sess=sess
        self.image_tensor=image_tensor
        self.detection_boxes=detection_boxes
        self.detection_scores=detection_scores
        self.detection_classes = detection_classes
        self.num_detections = num_detections
        self.category_index=category_index
        self.delay=1
        
        self.ret=False # video가 재생중인지 확인하기
        self.video_exist=False # video가 재생중인지 확인
        self.score_list=pd.DataFrame({ 'scores':[], 'classes':[]}) 
        self.result=pd.DataFrame({ 'scores':[], 'classes':[]})
        
        self.make_window() # window 만들기
        self.make_wiget() # wiget 만들기
        self.runModel()
        self.window.mainloop()
        
    def make_window(self):
        self.window=Tk()
        self.window.title("Ocean_Inspector")
        self.window.geometry("560x640")
        self.window.resizable(False,False)  # sizable
        #self.frame1=Frame(self.window,width=640,height=360, bg='green') # 이미지를 보여줄 곳
        
        #self.frame2=Frame(self.window,width=640,height=200,bg='blue') # reuslt 및 button 배치할 곳
        #self.frame1.pack()
        #self.frame2.pack()
    # make wiget
    def make_wiget(self):
        # frame 과 label의 관계

        self.resultButton=Button(self.window,text="Show Result" ,command=self.show_result,bg='blue')
        self.resultButton.pack(side='bottom')
        self.resultLabel=Label(self.window, text="result Label", bg='blue')
        self.resultLabel.pack(side='bottom')
        # imgLable에서 왜 문제가 발생하는지 찾아보기
        self.imgLabel=Label(self.window, width=640, height=360, text="image print",bg="red") # img label이 다른 것을 덮는다 : 이유를 알 수 없음 첫번째로 pack()하는 거랑 차이가 있는가
        self.imgLabel.pack(side='top')



        # 메뉴 구성하기
        menubar = Menu(self.window)  # window에 메뉴바 새성
        menu_1 = Menu(menubar, tearoff=0)
        menu_1.bind("<<MenuSelect>>")

        menu_1.add_command(label="Open", command=self.fileSelect)  # fileSelect function
        menubar.add_cascade(label="File", menu=menu_1)  # 클릭하면 "File" 뜬다
        self.window.config(menu=menubar)



    def fileSelect(self):
        selFile = filedialog.askopenfilename(initialdir="/", title='Select file',
                                             filetypes=(("Cideo files", "*.mp4;*.flv;*.avi;*.mkv"),
                                                        ("all files", "*.*")))  # 동
        self.video = cv2.VideoCapture(selFile) 
        self.video_exist=True

       
    # 지속적으로 threading
    def runModel(self):
        if self.video_exist:
            i=0
            while (self.video.isOpened()):

                ret,frame = self.video.read()
                print("---------------- frame----/----------------")
                frame_expanded = np.expand_dims(frame, axis=0)
                if len(frame_expanded.shape)!=4:break
                # frame 분석하기
                (boxes, scores, classes, num) = self.sess.run(  # self.sess
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: frame_expanded})

                
                assert len(scores)==len(classes) 
                #self.score_list['boxes']=boxes
                self.score_list['scores']=np.squeeze(scores)
                self.score_list['classes']=np.squeeze(classes)
                self.result=pd.concat([self.result,self.score_list])
          
                # Draw the results of the detection (aka 'visulaize the results')
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.80)
                print("---------------- frame--------------------")
                print("frame.shape : ", frame.shape)
                self.frame = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(frame)) #  frame 문제인거 같다
                i+=1
                self.ret=ret
                #if i==10: break

                self.stream() # self.imgLabel frame마다 업데이트 시키기
                  
                
                #self.stream()
            
           
        self.ret=False
        print("#######################동영상 종료#################################")
        print(self.result)
        print(self.result.shape)
        threading.Timer(self.delay,self.runModel).start() # runModel : self.delay 마다 지속적으로 실행

    # 동영상 self.imgLable 업데이트
    def stream(self):
        print("############## 재생여부#####################")
        print(self.ret)

        if self.ret:
            print("---------------------동영상 업데이트 시작------------------")
            print(self.frame)
            # button function이 끝나면 label이 업데이트 되는 점 유의
            self.imgLabel.config(image=self.frame) # 중지를 눌러야 뜬다
            self.imgLabel.image=self.frame
            print("---------------------동영상 업데이트 끝------------------")
            
    # 오염도 : frame당 쓰레기 
    def show_result(self): # 오염도 보여주기
        print("#####################result########################")
        self.frame_length=self.result.shape[0]
        self.result=self.result[self.result['classes']==1]
        self.pollutionScore=np.sum(self.result['scores'])/self.frame_length
        self.resultLabel.configure(text=str(self.pollutionScore*100)) # min-max 정규화하기
        print(np.sum(self.result['scores']))
        print(self.pollutionScore*100)

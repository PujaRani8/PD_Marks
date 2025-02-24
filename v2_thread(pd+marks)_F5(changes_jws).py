import os
import numpy as np
import cv2
import sys
import imutils
import gc
import math
import time
import re 
import shutil
import ctypes
from pyzbar.pyzbar import decode
import psycopg2
import pandas as pd
import onnxruntime as rt
from PySide2.QtWidgets import QCheckBox,  QApplication, QMainWindow, QLabel,QMessageBox ,QVBoxLayout,QHBoxLayout, QWidget, QLineEdit, QProgressBar, QPushButton, QFileDialog, QListWidget, QComboBox
from PySide2.QtGui import QFont 
from PySide2.QtCore import Qt, QThread, Signal
import webbrowser
import json
from ultralytics import YOLO
import contextlib
from threading import Thread
import traceback
import torch
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging





# from memory_profiler import profile


# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

#import logging
#from logging.handlers import TimedRotatingFileHandler

# # Define the log file directory and base name
# log_dir = "D:/path/to/logs"  # Replace with your desired path
# os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

# log_filename = os.path.join(log_dir, "application_log")  # Full path for the log file
# log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# # Set up a TimedRotatingFileHandler
# log_handler = TimedRotatingFileHandler(
#     filename=log_filename,
#     when='M',  # Rotate logs every 'M' minutes
#     interval=2,  # Interval for rotation (1 minute in this example) 
#     backupCount=24  # Keep the last 24 log files (optional)
# )

# log_handler.setFormatter(log_formatter)

# # Add handler to the root logger
# logging.getLogger().addHandler(log_handler)
# logging.getLogger().setLevel(logging.DEBUG)



########################## model called #####################################
script_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
onnx_model_number = os.path.join(script_dir, "Dependencies2", "number_noise_oct23.onnx")#"Number_noise_april9.onnx")
onnx_model_semester = os.path.join(script_dir, "Dependencies2", "semester_v3.onnx")
onnx_model_additional_pattern2 = os.path.join(script_dir, "Dependencies2", "additional_copy_pattern2.onnx")
onnx_model_semester_pattern2 = os.path.join(script_dir, "Dependencies2", "semester_modelv4.onnx")
onnx_model_sitting = os.path.join(script_dir, "Dependencies2", "sitting_v1.onnx")#### horizontal
onnx_model_sitting_vertical = os.path.join(script_dir, "Dependencies2", "sitting_vertical.onnx")###### vertical
onnx_model_additional = os.path.join(script_dir, "Dependencies2", "additional_noise_april9.onnx")
onnx_model_marks = os.path.join(script_dir, "Dependencies2","model_q1-q11_v7_no_aug.onnx")#model_q1-q11_v4.onnx")# "model_q1-q11_v5.onnx")#"model_q1-q11_v4.onnx")#"model_q1-q11_v3.onnx")  
onnx_model_total_marks = os.path.join(script_dir, "Dependencies2","pd_total_marks_aug8.onnx")# "model_pd_total_marks_noise12.onnx")
model_number = rt.InferenceSession(onnx_model_number)
model_semester = rt.InferenceSession(onnx_model_semester)
model_additional_pattern2 = rt.InferenceSession(onnx_model_additional_pattern2)
model_semester_pattern2 = rt.InferenceSession(onnx_model_semester_pattern2)
model_sitting = rt.InferenceSession(onnx_model_sitting)
model_sitting_vertical = rt.InferenceSession(onnx_model_sitting_vertical)
model_additional = rt.InferenceSession(onnx_model_additional)
model_marks = rt.InferenceSession(onnx_model_marks)
model_total_marks = rt.InferenceSession(onnx_model_total_marks)

###################################### html page for cordinates ##########################################
# html_path=os.path.join(script_dir, "Dependencies2","html_page_ru.html")
html_path=os.path.join(script_dir, "Dependencies2","html_page_pd_and_marks_old.html")

################# model for both pd and top anchor marks(detection model)##################################################
# model_path_pd =os.path.join(script_dir, "Dependencies2", "train9.pt") 
model_path_pd =os.path.join(script_dir, "Dependencies2", "train9_new3.pt") 

yolo_model_pd = YOLO(model_path_pd)

#################### model for marks top anchor(segmentation model)############################################
model_path_seg_top_anchor1 =os.path.join(script_dir, "Dependencies2", "marks_seg_anchor1.pt") 

marks_seg_anchor1 = YOLO(model_path_seg_top_anchor1) 

model_path_seg_top_anchor2 =os.path.join(script_dir, "Dependencies2", "marks_seg_anchor2.pt") 

marks_seg_anchor2 = YOLO(model_path_seg_top_anchor2) 


######################## model for marks bottom anchor#######################################################

model_path_ru =os.path.join(script_dir, "Dependencies2", "bottom_anchor_segment_model_F2.pt")#"bottom_anchor_segment_model.pt")#"RU_train8.pt")#"RU_train6.pt") 

yolo_model_ru = YOLO(model_path_ru)

model_path_ru_anchor1=os.path.join(script_dir, "Dependencies2", "ru_bottom_anchor1.pt") 
yolo_model_ru_anchor1=YOLO(model_path_ru_anchor1)

model_path_ru_anchor2=os.path.join(script_dir, "Dependencies2", "ru_bottom_anchor2.pt") 
yolo_model_ru_anchor2=YOLO(model_path_ru_anchor2)

######################################################################################################
model_path_barcode=os.path.join(script_dir, "Dependencies2", "barcode.pt") 
yolo_model_barcode=YOLO(model_path_barcode)

########################################## model for digit ocr ###########################################
yolo_path_digit=os.path.join(script_dir, "Dependencies2", "model_classification_F2.pt") 
yolo_digit=YOLO(yolo_path_digit)

yolo_path_digit_Q=os.path.join(script_dir, "Dependencies2", "marks_Q_digit_classification.pt") 
yolo_digit_Q=YOLO(yolo_path_digit_Q)

yolo_model_pd_digit_path=os.path.join(script_dir, "Dependencies2", "omr_digit_yolo_classification.pt") 
yolo_model_pd_digit=YOLO(yolo_model_pd_digit_path)

sam_checkpoint = os.path.join(script_dir, "Dependencies2", "marks_Q_digit_classification.pt") 

####################################### global variable ###################################################
input_folder_path_value = ""
output_folder_path_value = ""
template_path_value = ""
dbname_value = ""
table_name_value = ""
global SI_user
stop=0
error_result=0
total_image_length=0
roi_x=0
roi_y=0
roi_width=0
roi_height= 0
final_similarity_average=0
table_name=""
check_saved_folder='False'
###############################globally defined list #############################################
column_name_user11=[]
column_name_ocr1=[]
omr_type_user11=[]
question_list=[]
table_col_name=[]
column_list=[]
barcode_points11=[]
DOE_points=[]
Paper_code_points=[]
Paper_code_2_points=[]
Answer_copy_points=[]
center_code_points=[]
college_code_points=[]
Paper_No_points=[]
sitting_points=[]
sitting_vertical_points=[]
semester_points=[]
additional_points_pattern2=[]
semester_points_pattern2=[]
additional_points=[]
Subject_Code_points=[]
Q1_points=[]
Q2_points=[]
Q3_points=[]
Q4_points=[]
Q5_points=[]
Q6_points=[]
Q7_points=[]
Q8_points=[]
Q9_points=[]
Q10_points=[]
Q11_points=[]
total_points=[]
total_digit=[]      
Ansidrno_digit_cordinates=[]
Center_Code_digit_cordinates=[]
College_Code_digit_cordinates=[]
Paper_No_digit_coordinates=[]
Exam_digit_cordinates=[]
Paper_digit_cordinates=[]





def get_coordinate(image, first_anchor_cor1, second_anchor_cor1):  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_image = cv2.Canny(binary_image, threshold1=160, threshold2=255)
                
    fp = []
    sp = []
                
    for contour in cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
                    
        if first_anchor_cor1:
            x1, y1, w1, h1 = first_anchor_cor1

            # cropped_150_left = image[int(y1-10):int(h1+10), int(x1-10):int(w1+10)]
            # cropped_area_path_left= os.path.join(left_anchor_folder, base_filename)
            # cv2.imwrite( cropped_area_path_left ,cropped_150_left)

            if x1 - 10 < x < w1 + 10 and y1 - 10 < y < h1 + 10:  # left anchor area
                area = cv2.contourArea(contour)
            
                if 100 < area < 750:
                    #image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    fp = [x, y]    
                    
        if second_anchor_cor1:
            x2, y2, w2, h2 = second_anchor_cor1

            # cropped_150_right = image[int(y2-10):int(h2+10), int(x2-10):int(w2+10)]
            # cropped_area_path_right= os.path.join(right_anchor_folder, base_filename)
            # cv2.imwrite( cropped_area_path_right ,cropped_150_right)

            if x2 - 10 < x < w2 + 10 and y2 - 10 < y < h2 + 10:  # Right anchor area
                area = cv2.contourArea(contour)

                if 100 < area < 750:
                    #image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    sp = [x, y]
    del gray_image,  binary_image , canny_image,   first_anchor_cor1, second_anchor_cor1,image  
    
   
    return fp, sp 



def ru_sementation_model_for_anchor(image,model):
    #print("ru_sementation_model_for_anchor callled")
    results = model.predict(
        source=image,
        save=False,
        save_txt=False,
        save_crop=False,
        line_thickness=1,
        box=False,
        stream=False,verbose=False
    )
    data2 = []
    for result in results:
        boxes1 = result.boxes.numpy()
        data2.extend(boxes1.data.tolist())  # Accumulate data from all results
    fp=[]
    fp_shape=[]
    sp=[] 
    sp_shape=[]
     
    fp_width,sp_width=0,0
    fp_height,sp_height=0,0
    #print("data2 ===",data2)
    for data in data2:
        x, y, w, h, _, label = map(int, data[:6])
        if label == 0:
            #path = os.path.join(cropped_path1, base_filename)
            fp = [x, y]   
            fp_width=w
            #fp_height=h
            
        elif label == 1:
            #path = os.path.join(cropped_path2,base_filename)
            sp = [x, y] 
            sp_width=w    
            #sp_height=h
        
    return fp,sp,fp_width,sp_width

def ru_sementation_model_for_only_one_anchor(image,model):
    results = model.predict(
        source=image,
        save=False,
        save_txt=False,
        save_crop=False,
        line_thickness=1,
        box=False
    )
    data2 = []
    for result in results:
        boxes1 = result.boxes.numpy()
        data2.extend(boxes1.data.tolist())  # Accumulate data from all results
    cordinates=[]  
    # cordinates_width=[]  
    cordinates_width=0
    for data in data2:
        x, y, w, h, _, label = map(int, data[:6])
        cordinates = [x, y]
        cordinates_width=w
    return cordinates,cordinates_width


def find_anchor(cropped_150, yolo_model):
    first_anchor_cor = []
    second_anchor_cor = []
    results = yolo_model.predict(source=cropped_150, stream=False,verbose=False)#,save=True,save_txt=True,save_crop=True)
    #results = yolo_model.predict(source=cropped_150 , save=True,save_txt=True,save_crop=True,line_thickness= 1,box=False)
    for result in results:
        boxes1 = result.boxes.numpy()
        data1 = boxes1.data  # This line overwrites data1 in each iteration
        # Append data1 to a list if you want to accumulate data from all results
    data2 = data1.tolist()  # This line should be outside the loop if you want to accumulate data
    
    for data in data2:
        list = data
       
        if list[5] == 0:
            first_anchor_cor.append(list)
        else:
            second_anchor_cor.append(list)

    min_x_sublist = None  # Initialize to None
    
    try:
    
        min_x_sublist = max(first_anchor_cor, key=lambda box: box[4])
        first_anchor_cor1 = min_x_sublist[0:4]
    except ValueError:
        # Handle the case where first_anchor_cor is empty
        first_anchor_cor1 = None
    
    max_x_sublist = None  # Initialize to None
    
    try:
        
        max_x_sublist = max(second_anchor_cor, key=lambda box: box[4])
        second_anchor_cor1 = max_x_sublist[0:4]
    except ValueError:
        # Handle the case where second_anchor_cor is empty
        second_anchor_cor1 = None
    
    del results, data1, boxes1, data2, cropped_150, yolo_model, first_anchor_cor, second_anchor_cor
    return first_anchor_cor1, second_anchor_cor1


def rotate_pd(org_image,yolo_model,Image_Width,Image_Height):
    cropped_150 = org_image[30:160, :]##### original---150
    org_image= org_image[30:, :]
    first_anchor_cor1, second_anchor_cor1 = find_anchor(cropped_150, yolo_model)           
    if first_anchor_cor1 and second_anchor_cor1:
        fp, sp = get_coordinate(org_image.copy(), first_anchor_cor1, second_anchor_cor1 )          
        if fp and sp:
            x_crop=fp[0]
            y_crop=fp[1] 
            angle = math.degrees(math.atan2(sp[1] - fp[1], sp[0] - fp[0]))      
            rotated = imutils.rotate(org_image, angle)   
            cropped_image=rotated[ y_crop-20:,x_crop-30:]
            height,width=cropped_image.shape[:2]
            if width>=100 and height>=100:              
                cropped_image=cv2.resize(cropped_image,(Image_Width,Image_Height))
                final_cropped=cropped_image
            else:
                 #print("size of image is less than 100 after rotation")
                 final_cropped= None        
        else:
            final_cropped=None     
    else:
        final_cropped=None 
    return final_cropped

filename_done_by_segmentation=[]

def rotate_marks(org_image,yolo_model,image_width,Image_Width,Image_Height,base_filename):
    #print(" filename",base_filename)
    cropped_150 = org_image[30:160, :] ##### original---150
    org_image= org_image[30:, :]
    #print("first_anchor callled")
    first_anchor_cor1, second_anchor_cor1 = find_anchor(cropped_150, yolo_model)
    if first_anchor_cor1 and second_anchor_cor1:  ## anchor found by yolo ad  contour detected 
        x2, y2, w2, h2 = second_anchor_cor1 
        fp, sp = get_coordinate(org_image.copy(), first_anchor_cor1, second_anchor_cor1)
    else:
        fp=[]
        sp=[]
        
    if fp and sp:
        x_crop=fp[0]
        y_crop=fp[1] 
        angle = math.degrees(math.atan2(sp[1] - fp[1], sp[0] - fp[0]))      
        rotated = imutils.rotate(org_image, angle)
        crop_width=image_width-10
        cropped_image=rotated[y_crop-5:,x_crop-60:crop_width] 
        height,width=cropped_image.shape[:2]
        if width>=100 and height>=100:            
            cropped_image=cv2.resize(cropped_image,(Image_Width,Image_Height))
            final_cropped=cropped_image
        else:
            final_cropped= None   
    else:
        cropped_150_height,cropped_150_width,_=cropped_150.shape
        cropped_150_width=int(cropped_150_width/2)
        cropped_150_first_half=cropped_150[:, :cropped_150_width]
        cropped_150_second_half=cropped_150[:, cropped_150_width:]

        fp1,fp_width=ru_sementation_model_for_only_one_anchor(cropped_150_first_half,marks_seg_anchor1)
        sp1,sp_width=ru_sementation_model_for_only_one_anchor(cropped_150_second_half,marks_seg_anchor2)
        if fp1 and sp1:
            final_fp=[int(x) for x in fp1]
            final_sp=[int(x) for x in sp1] 
            final_sp[0]+= cropped_150_width
            filename_done_by_segmentation.append(base_filename)
            x_crop_seg=final_fp[0]
            y_crop_seg=final_fp[1] 
            angle = math.degrees(math.atan2(final_sp[1] - final_fp[1], final_sp[0] - final_fp[0]))      
            rotated = imutils.rotate(org_image, angle)
            crop_width_seg=image_width-10
            cropped_image=rotated[y_crop_seg-5:,x_crop_seg-60:crop_width_seg] 
            height,width=cropped_image.shape[:2]
            if width>=100 and height>=100:            
                cropped_image=cv2.resize(cropped_image,(Image_Width,Image_Height))
                final_cropped=cropped_image
            else:
                #print("size of image is less than 100 after rotation")
                final_cropped= None   
        else:
            final_cropped=None 
    return final_cropped



def rotate_marks_ru(org_image,image_width,yolo_model_ru,base_filename,Image_Width,Image_Height ):
    final_fp=[]
    final_sp=[]

    image_height,image_width,_=org_image.shape
    cropped_150 = org_image[-130:-20, :]##### original---150
    cropped_150_height,cropped_150_width,_=cropped_150.shape
    cropped_150_width=int(cropped_150_width/2)
    # crop_path=os.path.join(cropped_path1111,base_filename)
    # cv2.imwrite(crop_path,cropped_150)
    org_image_croped_height=image_height-130
    fp,sp,fp_width,sp_width= ru_sementation_model_for_anchor(cropped_150,yolo_model_ru)
    print(" first time sp_width" , sp_width)
    if fp:    
        if fp[0] >=70:
            print(" for fp ru_sementation_model_for_only_one_anchor called ")
            cropped_150_first_half=cropped_150[:, :cropped_150_width]
            fp1,fp_width= ru_sementation_model_for_only_one_anchor(cropped_150_first_half,yolo_model_ru_anchor1)
            print("ru_sementation_model_for_only_one_anchor fp",fp)
            final_fp=fp1
        else:
            final_fp=fp
    else:
        cropped_150_first_half=cropped_150[:, :cropped_150_width]
        fp1,fp_width= ru_sementation_model_for_only_one_anchor(cropped_150_first_half,yolo_model_ru_anchor1)
        final_fp=fp1

    if sp:
        if sp[0]<=cropped_150_width:
            cropped_150_second_half=cropped_150[:, cropped_150_width:]
            sp1,sp_width= ru_sementation_model_for_only_one_anchor(cropped_150_second_half,yolo_model_ru_anchor2)
            if sp1:
                sp1[0]= int(sp1[0])+cropped_150_width  
                sp_width=int(sp_width)+cropped_150_width 
            final_sp=sp1
        else:
            final_sp=sp
    else:
        cropped_150_second_half=cropped_150[:, cropped_150_width:]
        sp1,sp_width= ru_sementation_model_for_only_one_anchor(cropped_150_second_half,yolo_model_ru_anchor2)
        if sp1:
            sp1[0]= int(sp1[0])+cropped_150_width
            sp_width=int(sp_width)+cropped_150_width 
        final_sp=sp1

    
    if final_fp and final_sp:
        final_fp=[int(x) for x in final_fp]
        final_sp=[int(x) for x in final_sp]
        #sp_shape=[int(x) for x in sp_shape]

        final_fp[1]+= org_image_croped_height   
        final_sp[1]+= org_image_croped_height 

        x_crop=final_fp[0]+10
        #x_crop=fp_shape_width
        y_crop=final_fp[1] 
        angle = math.degrees(math.atan2(final_sp[1] - final_fp[1], final_sp[0] - final_fp[0]))   
        if angle >=-1 and angle <=30:
            rotated = imutils.rotate(org_image, angle)
            #cropped_image=rotated[:y_crop,x_crop:image_width-x_crop]
            cropped_image=rotated[20:y_crop,x_crop:sp_width+2]
            height,width=cropped_image.shape[:2]
      
            if width>=100 and height>=100:
                                
                cropped_image=cv2.resize(cropped_image,(Image_Width,Image_Height))
                final_cropped=cropped_image
            else:
                 final_cropped= None 
        else:
            final_cropped= None 
    else:
        final_cropped= None 

    return final_cropped

def straighten_image(input_path,yolo_model_pd,omr_type,Image_Width,Image_Height):

    base_filename = os.path.basename(input_path)
    org_image = cv2.imread(input_path) 
    image_height,image_width,_=org_image.shape

    if 'PD'in omr_type:
  
        final_cropped=rotate_pd(org_image,yolo_model_pd,Image_Width,Image_Height)
                    
    if 'Marks_Anchor_Top'in omr_type:
        print("calling Marks_Anchor_Top ") 
        final_cropped= rotate_marks(org_image,yolo_model_pd,image_width,Image_Width,Image_Height,base_filename)
                  
    if 'Marks_Anchor_Bottom'in omr_type:
        final_cropped= rotate_marks_ru(org_image,image_width,yolo_model_ru,base_filename,Image_Width,Image_Height )
    del  base_filename,org_image 
 
    return final_cropped

def straighten_image_ru_marks(input_path,template_image_width,Image_Width,Image_Height):
    base_filename = os.path.basename(input_path)
    org_image = cv2.imread(input_path) 
    image_height,image_width,_=org_image.shape
    final_cropped= rotate_marks_ru(org_image,image_width,yolo_model_ru,base_filename,Image_Width,Image_Height)
   
    if  final_cropped is not None:
       
        height,width=final_cropped.shape[:2]
        width_difference=template_image_width-width
        
        if width>=100 and height>=100:
                                
            if width_difference<=5:
                final_cropped=final_cropped
            else:
                final_cropped= None
                
        else:
            final_cropped= None
    del  base_filename,org_image 
    return final_cropped

 
class WorkerThread(QThread):
        finished = Signal(int)
        update_progress=Signal(int)
        db_error=Signal(str)
       # timer_one_image=Signal(float)
        omr_remaining_time=Signal(str)
        end_timer=Signal(str)
        #print("inside worker thread")

        def __init__(self, script_launcher):
            global  folder_path,template_path,table_name,db_name,user_name,password_db,host_name,port_name,template_image_height,\
               template_image, template_image_width, admit_card_available,admit_card_table_name,Image_Width,Image_Height,SI
            super().__init__()
            global  file_start_range,file_end_range
            self.script_launcher = script_launcher

            db_name=self.script_launcher.db_name
            user_name=self.script_launcher.user_name
            password_db=self.script_launcher.password_db
            host_name=self.script_launcher.host_name
            port_name=self.script_launcher.port_name
            table_name=self.script_launcher.table_name

            img_width = self.script_launcher.image_Width.text()
            img_height = self.script_launcher.image_Height.text()
            Image_Width=int(img_width) 
            Image_Height=int(img_height)

            similarity_per_value1 = self.script_launcher.similarity_per.text()
            SI_user = similarity_per_value1
            SI=int(SI_user)

            admit_card_table_name= self.script_launcher.admit_card_table_name
            if admit_card_table_name!='':
                admit_card_available=1
            else:
                admit_card_available=0
            template_path11=self.script_launcher.template_path
            template_image=cv2.imread(template_path11)
            template_image_height,template_image_width,_=template_image.shape
            folder_path=self.script_launcher.folder_path
            
            file_range_start_index =self.script_launcher.image_start.text()
            file_range_end_index=self.script_launcher.image_end.text()

            if check_saved_folder=='True':
                file_end_range=0
                file_start_range=0
                 
            else:
                file_end_range=int(file_range_end_index)
                file_start_range=int(file_range_start_index)
     
        def run(self):

            global column_name_user11, folder_path,SI,final_similarity_average,table_name,column_list,omr_type_user11
            global error_result,column_name_user,omr_type_user,pd_column_name_admit_card,ocr_column,column_name_ocr1,omr_type_list1 
           
            column_name_user1 = self.script_launcher.columns_platform.text()
            column_name_user11.append(column_name_user1)
            column_name_user=column_name_user11
            column_name_user11=[]
            omr_type_user1= self.script_launcher.columns_platform_question.currentText()
            omr_type_user11.append(omr_type_user1)
            omr_type_user=omr_type_user11
            omr_type_user11=[]

            ocr_column = self.script_launcher.columns_platform_ocr.text()
            column_name_ocr1.append(ocr_column)
            pd_column_name_admit_card=column_name_ocr1
            ocr_column =[]
            column_name_ocr1=[]
            first_string = column_name_user[0]
            column_list = [item.strip() for item in first_string.split(',')]
            first_string = omr_type_user[0]
            omr_type_list1 = [item.strip() for item in first_string.split(',')]
            self.corrdinates_points_process()
            start = time.time()
            self.omr_check_function()
            end = time.time()
            total_time=end-start
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            total_time_hours=f'{int(hours)} hr {int(minutes)} min {int(seconds)} sec'
            self.end_timer.emit(total_time_hours)
            self.finished.emit(error_result)

  
        def corrdinates_points_process(self):
            global roi_x,roi_y,roi_width,roi_height,barcode_points11,semester_points_pattern2,\
             DOE_points,Paper_code_points,Paper_code_2_points,Answer_copy_points,additional_points_pattern2,\
            center_code_points,college_code_points,Paper_No_points,sitting_points,sitting_vertical_points,\
                semester_points,additional_points,omr_type_list1,Subject_Code_points,\
            Q1_points,Q2_points,Q3_points,Q4_points,Q5_points,Q6_points,Q7_points,Q8_points,Q9_points,Q10_points,\
                Q11_points,total_points,total_digit

            first_string = column_name_user[0]
            column_list = [item.strip() for item in first_string.split(',')]
           
            first_string = omr_type_user[0]
            omr_type_list1 = [item.strip() for item in first_string.split(',')]
     
            if 'Barcode' in column_list:
                barcode_points11=barcode_points11_user 
                roi_x, roi_y, roi_width, roi_height = barcode_points11[0] 
           
            if 'Ansidrno' in column_list:
                Answer_copy_points=Answer_copy_points_user
   
            if 'Additional_Copy' in column_list: 
                additional_points  =additional_points_user

            if 'Center_Code' in column_list:
                center_code_points=  center_code_points_user
            
            if 'College_Code' in column_list:
                college_code_points=  college_code_points_user
            
            if 'Paper_No' in column_list:
                Paper_No_points= Paper_No_points_user

            if 'Date_of_Exam' in column_list:
                   DOE_points=   DOE_points_user

            if 'Paper_Code' in column_list:
                Paper_code_points=Paper_code_points_user

            if 'Paper_Code_2' in column_list:
                Paper_code_2_points=Paper_code_2_points_user

            if 'Semester' in column_list:
                semester_points=semester_points_user

            if 'Additional_Copy_Pattern2' in column_list: 
                additional_points_pattern2  =additional_points_user_pattern2
            
            if 'Semester_Pattern2' in column_list:
                semester_points_pattern2=semester_points_user_pattern2
            
            if 'Sitting_Horizontal' in column_list:
                sitting_points=sitting_points_user

            if 'Sitting_Vertical' in column_list:
                sitting_vertical_points=sitting_vertical_points_user
             
            if 'Subject_Code' in column_list:
                Subject_Code_points=Subject_Code_user

            
             
            if 'Q1' in column_list:
                Q1_points=Q1_points_user

            if 'Q2' in column_list:
                Q2_points=Q2_points_user
      
            if 'Q3' in column_list:
                Q3_points=Q3_points_user
          

            if 'Q4' in column_list:
                Q4_points=Q4_points_user

            if 'Q5' in column_list:
                Q5_points=Q5_points_user
        
            if 'Q6' in column_list:
                Q6_points=Q6_points_user

            if 'Q7' in column_list:
                Q7_points=Q7_points_user

            if 'Q8' in column_list:
                Q8_points=Q8_points_user

            if 'Q9' in column_list:
                Q9_points=Q9_points_user

            if 'Q10' in column_list:
                Q10_points=Q10_points_user

            if 'Q11' in column_list:
                Q11_points=Q11_points_user

            if 'Total_Marks' in column_list:
                total_points=total_points_user

            if "Total_Marks_OCR" in column_list:
                total_digit= total_digit_user 

         
###########################omr check ##############################################################            
        def omr_check_function(self):
            global roi_x,roi_y,roi_width,roi_height,barcode_points11,additional_points_pattern2,\
                    DOE_points,Paper_code_points,Paper_code_2_points,Answer_copy_points,semester_points_pattern2,\
                    center_code_points,college_code_points,Paper_No_points,sitting_points,semester_points,additional_points,\
                    Q1_points,Q2_points,Q3_points,Q4_points,Q5_points,Q6_points,Q7_points,Q8_points,Q9_points,Q10_points,Q11_points,total_points
                
 
            additional_mapping={12:'Additional_Copy_1',13:'Additional_Copy_2', 14:'-', 15:'*'}
            number_mapping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10:'*',11:'-'}
            semester_mapping={1:"sem1",2:"sem2",3:"sem3",4:"sem4",5:"sem5",6:"sem6",7:"sem7",8:"sem8",9:'*',10:"-"}
            sitting_mapping={11:"1st_sitting",12:"2nd_sitting",13:"-",14:"*"}
            pd_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '*', 11: '-'}
            additional_mapping_pattern2={1:'Additional_Copy_1',2:'Additional_Copy_2', 0:'-', 3:'*'}
            semester_mapping_pattern2={1:"sem1",2:"sem2",3:"sem3",4:"sem4",5:"sem5",6:"sem6",7:"sem7",8:"sem8",9:'sem9',10:"sem10",0:"-",11:"*"}
            batch_size = 1
            suffix = 1
            while True:
                if suffix == 1:
                    unmatched_folder = os.path.join(folder_path, "unmatched")
                    straightened_folder = os.path.join(folder_path, "final_straightened_image")
                    os.makedirs(unmatched_folder, exist_ok=True)
                    os.makedirs(straightened_folder, exist_ok=True)
                    break
                suffix += 1

            
            def prediction_classification_pd(image_path, x, y, width,height,yolo_model_pd_digit):
                image = cv2.imread(image_path)
                cropped_image = image[y:y + height, x:x + width]
                results = yolo_model_pd_digit(cropped_image,verbose=False) 
                name_dict=results[0].names  
                prob1=results[0].probs 
                label_top=prob1.top1 
                confidence=prob1.top1conf.numpy()
                confidence_top=confidence.tolist() 
                if  label_top is None: 
                        roll_label="*"
                elif label_top==10:
                    roll_label="-"
                else:
                    roll_label=str(label_top)
                del cropped_image,image
                return roll_label

            def prediction_classification_marks(image_path, x, y, width,height,yolo_digit):
                image = cv2.imread(image_path)

                cropped_image = image[y:y + height, x:x + width]
                
                base_filename = os.path.basename(image_path)
                #digit_path = os.path.join(slice_folder,  os.path.join(slice_folder, f'{base_filename}_{x}_{y}.jpg'))
               # cv2.imwrite(digit_path, cropped_image)

                results = yolo_digit(cropped_image,verbose=False) 
                name_dict=results[0].names  
                prob1=results[0].probs 
                label_top=prob1.top1 
                confidence=prob1.top1conf.numpy()
                confidence_top=confidence.tolist() 
                if  label_top is None: 
                        roll_label="*"
                elif label_top==10:
                    roll_label="-"
                else:
                    roll_label=str(label_top)
                del cropped_image,image
                return roll_label
            
            def prediction_classification_Q(image_path, x, y, width,height,yolo_digit):
                image = cv2.imread(image_path)
                cropped_image = image[y:y + height, x:x + width]               
                base_filename = os.path.basename(image_path)
                #digit_path = os.path.join(slice_folder_Q,  os.path.join(slice_folder_Q, f'{base_filename}_{x}_{y}.jpg'))
                #cv2.imwrite(digit_path, cropped_image)
                results = yolo_digit(cropped_image,verbose=False) 
                name_dict=results[0].names  
                prob1=results[0].probs 
                label_top=prob1.top1 
                confidence=prob1.top1conf.numpy()
                confidence_top=confidence.tolist() 
                if label_top is None: 
                        roll_label="*"
                elif label_top==10:
                    roll_label="-"
                else:
                    roll_label=str(label_top)
                del cropped_image,image
                return roll_label
            
            def crop_top_half(image,roi_x,roi_y,roi_width,roi_height):
                cropped_image = image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                return cropped_image
                    
            def read_barcode(image,x,y,w,h,image_path):#### for barcode detection using pyzbar
                roi=image[y:h, x:w]
                barcodes = decode(roi)
                barcode_data = ''
                for barcode in barcodes:
                    barcode_data += barcode.data.decode('utf-8')  
                return barcode_data

            
            def model_yolo_prediction(croped_image,yolo_model_barcode):  #### for barcode detection using yolo          
                data2=[]    
                barcode_list=[]  
                results=yolo_model_barcode.predict(source=croped_image,box=False,stream=True, verbose=False)
                
                for result in results:
                    boxes1 = result.boxes.numpy()
                    data3 = boxes1.data 
                    if boxes1 is None or len(boxes1) == 0:
                        return 0,0,0,0
                    else: 
                        data1 = boxes1.data 
                        data2 = data1.tolist() 
                for data in data2:
                    list = data            
                    x=int(list[0])
                    y=int(list[1])
                    w=int(list[2])
                    h=int(list[3])
                    conf_score=list[4]
                    data_list=[x,y,w,h,conf_score]
                    barcode_list.append(data_list)         
                if len(barcode_list)>1:
                    barcode_cordinates = max(barcode_list, key=lambda item: item[4])
                    bcor_x=barcode_cordinates[0]
                    bcor_y=barcode_cordinates[1]
                    bcor_w=barcode_cordinates[2]
                    bcor_h=barcode_cordinates[3]
                    return bcor_x,bcor_y,bcor_w,bcor_h
                else:
                    barcode_cordinates=barcode_list[0]            
                    bcor_x=barcode_cordinates[0]
                    bcor_y=barcode_cordinates[1]
                    bcor_w=barcode_cordinates[2]
                    bcor_h=barcode_cordinates[3]
                    return bcor_x,bcor_y,bcor_w,bcor_h
                
            def model_yolo(image_path,yolo_model_barcode,roi_x,roi_y,roi_width,roi_height):  ##### yolo for barcode position detection
                image = cv2.imread(image_path)
                img_height,image_width,_=image.shape
                croped_image=crop_top_half(image,roi_x,roi_y,roi_width,roi_height)
                barcode_x,barcode_y,barcode_w,barcode_h=model_yolo_prediction(croped_image,yolo_model_barcode)
                if barcode_x!=0 and barcode_y!=0:
                    barcode_result=read_barcode(croped_image,barcode_x,barcode_y,barcode_w,barcode_h,image_path)
                    if barcode_result == '':
                            x_axis=0
                            y_axis=0
                            x_width=image_width
                            y_height=img_height
                            barcode_result1=read_barcode(croped_image,x_axis,y_axis,x_width,y_height,image_path)# direct function call if yolo not detected
                            barcode_blank=''
                            if barcode_result1 == '':
                                return barcode_blank
                            
                            else:
                                return barcode_result1          
                    else:
                        return barcode_result
                else:
                    ######### 'else' part  will call when yolo fails to detect barcode cordinates area (barcode_x=0 and barcode_y=0)
                    x_axis=0
                    y_axis=0
                    x_width=image_width
                    y_height=img_height
                    barcode_result1=read_barcode(croped_image,x_axis,y_axis,x_width,y_height,image_path)# direct function call if yolo not detected                 
                    barcode_blank=''
                    if barcode_result1 == '':
                        return barcode_blank
                    else:
                        return barcode_result1  
            def predict_common(image_path, x, y, width, height,model,mapping,resize_width,resize_height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(resize_width ,resize_height)) 
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1) 
                input_name = model.get_inputs()[0].name
                output_name = model.get_outputs()[0].name
                prediction = model.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = mapping.get(predicted_class, '')
                del resized_image,cropped_image,image,image_array
                return predicted_label
            def predict_Additional_Patern2(image_path, x, y, width, height):
                image = cv2.imread(image_path)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(128 ,128))    # (280 ,40)
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                input_name = model_additional_pattern2.get_inputs()[0].name
                output_name = model_additional_pattern2.get_outputs()[0].name
                prediction = model_additional_pattern2.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = additional_mapping_pattern2.get(predicted_class, '')
                del resized_image,cropped_image,image,image_array
                return predicted_label

            def process_barcode(image_path):
                barcode_data=model_yolo(image_path,yolo_model_barcode,roi_x,roi_y,roi_width,roi_height)
                return barcode_data
            
            def process_ansidrno(image_path):
                answer_copy = ''
                for point in Answer_copy_points:
                    x, y, answer_copy_width, answer_copy_height = point
                    answer_copy += predict_common(image_path, x, y, answer_copy_width, answer_copy_height,model_number,number_mapping,50,250)
                return answer_copy
            def process_additional_copy(image_path):
                additional_copy=''
                for point in additional_points:
                    x, y, additional_copy_width, additional_copy_height = point
                    additional_copy+= predict_common(image_path, x, y, additional_copy_width, additional_copy_height,model_additional,additional_mapping,280 ,40)
                return additional_copy
            
            def process_center_code(image_path):
                center_code = ''

                for point in center_code_points:
                    x, y, center_code_width, center_code_height = point
                    center_code += predict_common(image_path, x, y, center_code_width, center_code_height,model_number,number_mapping,50,250)
                return center_code
            def process_college_code(image_path):
                college_code = ''

                for point in college_code_points:
                    x, y, college_code_width, college_code_height = point
                    college_code += predict_common(image_path, x, y, college_code_width, college_code_height,model_number,number_mapping,50,250)
                return college_code  
            def process_paper_no(image_path):
                paper_no = ''

                for point in Paper_No_points:
                    x, y, paper_no_width, paper_no_height = point
                    paper_no += predict_common(image_path, x, y, paper_no_width, paper_no_height,model_number,number_mapping,50,250)
                return paper_no
            
            def process_doe(image_path):
                doe = ''
                for point in DOE_points:
                    x, y, doe_width, doe_height = point
                    doe += predict_common(image_path, x, y, doe_width, doe_height,model_number,number_mapping,50,250)
                return doe
            
            def process_paper_code(image_path):
                paper_code = ''
                for point in Paper_code_points:
                    x, y,paper_code_width, paper_code_height = point
                    paper_code += predict_common(image_path, x, y, paper_code_width, paper_code_height,model_number,number_mapping,50,250)
                return paper_code
            
            def process_subject_code(image_path):
                subject_code = ''
                for point in Subject_Code_points:
                    x, y,subject_code_width, subject_code_height = point
                    subject_code += predict_common(image_path, x, y, subject_code_width, subject_code_height,model_number,number_mapping,50,250)
                return subject_code
            def process_paper_code2(image_path):
                paper_code_2 = ''
                for point in Paper_code_2_points:
                    x, y,paper_code_width, paper_code_height = point
                    paper_code_2 += predict_common(image_path, x, y, paper_code_width, paper_code_height,model_number,number_mapping,50,250)
                paper_code_2_alpha=paper_code_2[-1]
                paper_code_2=paper_code_2[:-1]
                if paper_code_2_alpha=="*" or paper_code_2_alpha=="-" or  paper_code_2_alpha=="--" :
                    paper_code_2+=paper_code_2_alpha
                else:
                    paper_code_2_alpha=int(paper_code_2_alpha)
                    alpha_value=chr(paper_code_2_alpha+65)
                    paper_code_2+=alpha_value
                return paper_code_2
            def process_semester(image_path):
                semester=''
                for point in semester_points:
                    x, y, semester_width, semester_height = point
                    semester += predict_common(image_path, x, y, semester_width, semester_height,model_semester,semester_mapping,280 ,40)
                return semester
            def process_additional_copy_pattern2(image_path):
                additional_copy=''
                for point in additional_points_pattern2:
                    x, y, additional_copy_width, additional_copy_height = point
                    additional_copy+= predict_Additional_Patern2(image_path, x, y, additional_copy_width, additional_copy_height)
                return additional_copy
            
            def process_semester_pattern2(image_path):
                semester=''
                for point in semester_points_pattern2:
                    x, y, semester_width, semester_height = point
                    semester += predict_common(image_path, x, y, semester_width, semester_height,model_semester_pattern2,semester_mapping_pattern2,128,128)
                return semester
            def process_sitting_horizontal(image_path):
                sitting=''
                for point in sitting_points:
                    x, y, sitting_width, sitting_height = point
                    sitting += predict_common(image_path, x, y, sitting_width, sitting_height,model_sitting,sitting_mapping,160,40)
                return sitting
            def process_sitting_vertical(image_path):
                sitting_v=''
                for point in sitting_vertical_points:
                    x, y, sitting_width, sitting_height = point
                    sitting_v += predict_common(image_path, x, y, sitting_width, sitting_height,model_sitting_vertical,sitting_mapping,60,240)
                return sitting_v
            ''' This function takes in a single image and process it and returns the data'''
           
            #@profile
            def process_pd_column(column_name, image_path, result_row):

                if column_name == 'Ansidrno':
                    result_row['Ansidrno'] = process_ansidrno(image_path)
                elif column_name == 'Additional_Copy':
                    result_row['Additional_Copy'] = process_additional_copy(image_path)
                elif column_name == 'Center_Code':
                    result_row['Center_Code'] = process_center_code(image_path)
                elif column_name == 'College_Code':
                    result_row['College_Code'] = process_college_code(image_path)
                elif column_name == 'Paper_No':
                    result_row['Paper_No'] = process_paper_no(image_path)
                elif column_name == 'Date_of_Exam':
                    result_row['Date_of_Exam'] = process_doe(image_path)
                elif column_name == 'Paper_Code':
                    result_row['Paper_Code'] = process_paper_code(image_path)
                elif column_name == 'Subject_Code':
                    result_row['Subject_Code'] = process_subject_code(image_path)
                elif column_name == 'Paper_Code_2':
                    result_row['Paper_Code_2'] = process_paper_code2(image_path)
                elif column_name == 'Semester':
                    result_row['Semester'] = process_semester(image_path)
                elif column_name == 'Additional_Copy_Pattern2':
                    result_row['Additional_Copy_Pattern2'] = process_additional_copy_pattern2(image_path)
                elif column_name == 'Semester_Pattern2':
                    result_row['Semester_Pattern2'] = process_semester_pattern2(image_path)
                elif column_name == 'Sitting_Horizontal':
                    result_row['Sitting_Horizontal'] = process_sitting_horizontal(image_path)
                elif column_name == 'Sitting_Vertical':
                    result_row['Sitting_Vertical'] = process_sitting_vertical(image_path)

            #@profile
            def process_image_pd(image_path, column_list11, filename):
                result_row = {'Filename': os.path.basename(filename)}
                Scanno = os.path.splitext(os.path.basename(filename))[0]
                result_row['Scanno'] = Scanno

                if 'Barcode' in column_list11:
                    result_row['Barcode'] = process_barcode(image_path)

                # Use ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=3) as executor:  # Adjust max_workers as needed
                    futures = [
                        executor.submit(process_pd_column, column, image_path, result_row)
                        for column in column_list11
                    ]

                    # Wait for all threads to complete and handle exceptions if needed
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error processing column: {e}")

                gc.collect()  # Optional: To release memory
                return result_row
            
            def proces_all_Q(image_path, Q_points):
                value = ''
               
                for point in Q_points:
                    x, y, Q_width, Q_height = point
                    x_ocr = 1
                    y_ocr = y - 5
                    ocr_width = x
                    ocr_height = Q_height + 7
                    value += predict_common(image_path, x, y, Q_width, Q_height, model_marks, pd_mapping, 350, 50)
                return value
            
            def process_all_Q_digit(image_path,value, Q_points):
                value_list=list(value)
                check11 = 0
                for i,val in enumerate(value_list):
                    if '-' in val or '*' in val or val is None:
                            #print("val",val)
                            #print(f' Q_points[i]', Q_points[i])
  
                            x, y, Q_width, Q_height = Q_points[i]
                            x_ocr = 1
                            y_ocr = y - 5
                            ocr_width = x
                            ocr_height = Q_height + 7
                            ocr_val = prediction_classification_Q(image_path, x_ocr, y_ocr, ocr_width, ocr_height, yolo_digit_Q)
                            if '-' not in ocr_val and '*' not in ocr_val:
                                value_list[i]=ocr_val
                                check11 = 1
                            else:
                                value_list[i]=val                  
                    else:
                        value_list[i]=val
                
              
                blank_index_list = []

                # Identify the indices of dashes ('-') in the value
                for i, x in enumerate(value_list):
                    if x == '-':
                        blank_index_list.append(i)

                # If there are exactly 2 dashes, replace value_list with [0, 0]
                if len(blank_index_list) == 2:
                    value_list = ['0', '0']
                else:
                    value_list=value_list

                value_str= ''.join(value_list)
                return value_str

            def process_total_marks(image_path):
                total = ''
                for point in total_points:
                    x, y, total_width, total_height = point
                    total += predict_common(image_path, x, y, total_width, total_height, model_total_marks, number_mapping, 300, 30)
                return total

        
            def process_marks_column(image_path, column_list11, filename):
                result_row = {'Filename': os.path.basename(filename)}
                Scanno = os.path.splitext(os.path.basename(filename))[0]
                result_row['Scanno'] = Scanno
                ques_cord_dict = {
                    "Q1": Q1_points, "Q2": Q2_points, "Q3": Q3_points, "Q4": Q4_points,
                    "Q5": Q5_points, "Q6": Q6_points, "Q7": Q7_points, "Q8": Q8_points,
                    "Q9": Q9_points, "Q10": Q10_points, "Q11": Q11_points,
                }

                # Function to process a single question
                def process_question(question_num):
                    result_row[f'Q{question_num}'] = proces_all_Q(image_path, globals()[f'Q{question_num}_points'])

                # Function to process Total Marks
                def process_total_marks_thread():
                    result_row['Total_Marks'] = process_total_marks(image_path)

                with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust `max_workers` as needed
                    # Add tasks for questions
                    futures = [
                        executor.submit(process_question, i)
                        for i in range(1, 12) if f'Q{i}' in column_list11
                    ]

                    # Add task for Total Marks if it exists in the column list
                    if 'Total_Marks' in column_list11:
                        futures.append(executor.submit(process_total_marks_thread))

                    # Wait for all tasks to complete
                    for future in futures:
                        try:
                            future.result()  # Ensure task completion and catch errors
                        except Exception as e:
                            print(f"Error in threaded task: {e}")

                # Process each question digit correction
                for key in result_row:
                    if key in column_list11 and key in ques_cord_dict:
                        q_cordinates = ques_cord_dict[key]
                        result_row[key] = process_all_Q_digit(image_path, result_row[key], q_cordinates)

                # Process Barcode
                if 'Barcode' in column_list11:
                    result_row['Barcode'] = process_barcode(image_path)

                # Print or return final result
                return result_row


            
            # Main function to process image marks with threading support

            def process_image_marks(image_path, column_list11, filename):
                result_row = {'Filename': os.path.basename(filename)}
                
                # Start processing marks columns with threading support
                return process_marks_column(image_path, column_list11, filename)
            '''Saves the data in the designated table of postgres database'''
            def database_process(table_name, result_row,db_column_name):
                global error_result
                try:
                    # Connect to the database
                    conn = psycopg2.connect(
                        host=host_name,
                        port=port_name,
                        database=db_name,
                        user=user_name,
                        password=password_db
                    )

                    with conn:
                        with conn.cursor() as cursor:
                            column_list_DB = db_column_name 
                            columns = ['Filename'] + ['Scanno'] + column_list_DB  
                            # Create the table if it does not exist
                            column_definitions = [f'{col} TEXT' for col in columns]
                            create_table_sql = f'''
                                CREATE TABLE IF NOT EXISTS {table_name} (
                                    {', '.join(column_definitions)}
                                )
                            '''
                            cursor.execute(create_table_sql)
                            # Prepare the insert statement
                            placeholders = ', '.join(['%s'] * len(columns))
                            insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                            values = [result_row.get(column, '') for column in columns]

                            # Execute the insert statement
                            cursor.execute(insert_sql, values)

                except psycopg2.Error as e:
                    error_result = 1
                    db_connection_error = f"{e}"
                    print(f"Database error: {db_connection_error}")
                    exit_program()
#------------------- this for marks omr ----------------
            def database_process2(table_name, data, db_column_name):
                global error_result
                try:
                    # Connect to the database
                    conn = psycopg2.connect(
                        host=host_name,
                        port=port_name,
                        database=db_name,
                        user=user_name,
                        password=password_db
                    )

                    with conn:
                        with conn.cursor() as cursor:
                            # Construct the column list for the database table
                            columns = ['Filename'] + ['Scanno'] + db_column_name +["sum_question"]+ ['updation']+['Flag'] +["Total_Attempted_Column"]

                            # Create the table if it does not exist
                            column_definitions = [f'{col} TEXT' for col in columns]
                            create_table_sql = f'''
                                CREATE TABLE IF NOT EXISTS {table_name} (
                                    {', '.join(column_definitions)}
                                )
                            '''
                            cursor.execute(create_table_sql)

                            # Prepare the insert statement
                            placeholders = ', '.join(['%s'] * len(columns))
                            insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

                            # Iterate through the list of lists of dictionaries and insert data
                            for row in data:
                                for result_row in row:
                                    # Ensure Filename and Scanno are included in the row
                                    result_row['Filename'] = result_row.get('Filename', 'default_filename')
                                    result_row['Scanno'] = result_row.get('Scanno', 'default_scanno')
                                    
                                    # Generate values for insertion
                                    values = [result_row.get(column, '') for column in columns]
                                    cursor.execute(insert_sql, values)

                except psycopg2.Error as e:
                    error_result = 1
                    db_connection_error = f"{e}"
                    print(f"Database error: {db_connection_error}")
                    exit_program()
            


#------------------- this for pd omr ----------------

            #@profile
            def database_process3(table_name, data):
                global error_result
                print("Under process 3---------------")
                try:
                    # Connect to the database
                    conn = psycopg2.connect(
                        host=host_name,
                        port=port_name,
                        database=db_name,
                        user=user_name,
                        password=password_db
                    )
                    # print("Connection is established-------")
                    with conn:
                        with conn.cursor() as cursor:
                            # Extract db_column_name from the first non-empty list of tuples
                            db_column_name = None
                            for row_group in data:
                                if row_group:  # Check if row_group is not empty
                                    first_tuple = row_group[0]
                                    if first_tuple and len(first_tuple) > 1:
                                        db_column_name = first_tuple[1]
                                        break
                            
                            if not db_column_name:
                                print("Error: No valid db_column_name found in the data.")
                                return  # Exit the function if no column names are found

                            # print("db_column_name is established----------", db_column_name)

                            # Create the full list of columns, including Filename and Scanno
                            columns = ['Filename', 'Scanno'] + db_column_name 

                            # Create the table if it does not exist
                            column_definitions = [f'{col} TEXT' for col in columns]
                            create_table_sql = f'''
                                CREATE TABLE IF NOT EXISTS {table_name} (
                                    {', '.join(column_definitions)}
                                )
                            '''
                            cursor.execute(create_table_sql)

                            # Prepare the insert statement
                            placeholders = ', '.join(['%s'] * len(columns))
                            insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

                            # Iterate through the list of lists of tuples
                            # print("^^^^^^^^^^^^^^At for loop in database3!!!!!!!!!")
                            for row_group in data:
                                if not row_group:  # Skip empty row groups
                                    continue
                                for row in row_group:
                                    if not row or len(row) < 2:  # Skip invalid tuples
                                        continue

                                    row_data = row[0]  # First element in the tuple is the dictionary
                                    dynamic_columns = row[1]  # Second element in the tuple is the list of columns

                                    # Add default values for Filename and Scanno if not present
                                    row_data['Filename'] = row_data.get('Filename', 'default_filename')
                                    row_data['Scanno'] = row_data.get('Scanno', 'default_scanno')
                                    

                                    # Ensure that the values match the full list of columns
                                    values = [row_data.get(column, '') for column in columns]
                                    cursor.execute(insert_sql, values)

                except psycopg2.Error as e:
                    error_result = 1
                    db_connection_error = f"{e}"
                    print(f"Database error: {db_connection_error}")
                    exit_program()


            def exit_program():
                #print("Exiting the program...")
                sys.exit(0)

            ''' Staighten the images using SIFT'''
    
            #@profile
            def straighten_images_with_yolo(input_path, SI, count, batch_count, image_batches, yolo_model_pd, omr_type_list1):
                """
                Straightens images using yolo and saves the result or original image based on straightening success.
                
                Args:
                    input_path (str): Path to the input image.
                    SI, count, batch_count, image_batches: Unused parameters (part of the larger system context).
                    yolo_model_pd: YOLO model instance or configuration for processing.
                    omr_type_list1: List of OMR configurations or settings.
                
                Returns:
                    list: Paths of successfully straightened images.
                """
                straightened_image_paths = []
                base_filename = os.path.basename(input_path)
                # Attempt to straighten the image
                if  'Marks_Anchor_Bottom'in omr_type_list1:
                    straightened_image =straighten_image_ru_marks(input_path,template_image_width,Image_Width,Image_Height)
                    if straightened_image is not None:
                    # Perform template matching
                        similarity_value = cv2.matchTemplate(straightened_image, template_image, cv2.TM_CCOEFF_NORMED)
                        similarity_percentage = similarity_value.max() * 100
                        if similarity_percentage >=SI:
                            straightened_path = os.path.join(straightened_folder, base_filename)
                            cv2.imwrite(straightened_path, straightened_image)
                            straightened_image_paths.append(straightened_path)
                        else:
                            unmatched_path = os.path.join(unmatched_folder, base_filename)
                            cv2.imwrite(unmatched_path, cv2.imread(input_path))
                    else:
                        unmatched_path = os.path.join(unmatched_folder, base_filename)
                        cv2.imwrite(unmatched_path, cv2.imread(input_path))

                elif 'Marks_Anchor_Top'in omr_type_list1:
                    straightened_image =straighten_image(input_path, yolo_model_pd, omr_type_list1,Image_Width,Image_Height)
                    if straightened_image is not None:
                    # Perform template matching
                        similarity_value = cv2.matchTemplate(straightened_image, template_image, cv2.TM_CCOEFF_NORMED)
                        similarity_percentage = similarity_value.max() * 100
                        if similarity_percentage >=SI:
                           
                            straightened_path = os.path.join(straightened_folder, base_filename)
                            cv2.imwrite(straightened_path, straightened_image)
                            straightened_image_paths.append(straightened_path)
                        else:
                            
                            unmatched_path = os.path.join(unmatched_folder, base_filename)
                            cv2.imwrite(unmatched_path, cv2.imread(input_path))
                    else:
                        unmatched_path = os.path.join(unmatched_folder, base_filename)
                        cv2.imwrite(unmatched_path, cv2.imread(input_path))
                elif 'PD'in omr_type_list1:
                    straightened_image = straighten_image(input_path, yolo_model_pd, omr_type_list1,Image_Width,Image_Height)
                    if straightened_image is not None:
                        # Save the straightened image to the straightened folder
                        straightened_path = os.path.join(straightened_folder, base_filename)
                        cv2.imwrite(straightened_path, straightened_image)
                        straightened_image_paths.append(straightened_path)
                    else:
                        # Save the original image to the unmatched folder
                        unmatched_path = os.path.join(unmatched_folder, base_filename)
                        cv2.imwrite(unmatched_path, cv2.imread(input_path))    
                del straightened_image # Explicitly free memory               
                return straightened_image_paths


                # for input_path in image_paths:
           
            def is_scientific_notation(number_str):
                # Regular expression to match scientific notation
                scientific_notation_pattern = re.compile(r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$')
                return bool(scientific_notation_pattern.match(number_str))

            def count_value(count):
                 count+=1
                 return count

            def find_index(image_batch_list, file_number):
                for index, value in enumerate(image_batch_list):
                    if value[0] == file_number:
                        return index
                return -1  # Return -1 if not found     
            
            '''This function processes a batch of images and extracts OCR data for the total marks field.'''
            def digit_ocr(batch_straightened_images):
                digit_ocr={}
                for straightened_image in batch_straightened_images:
                    if straightened_image is not None :
                        digit = ''
                        for point in total_points:
                            x, y, total_width,total_height = point
                            x_ocr=1
                            y_ocr=y
                            ocr_width=x
                            ocr_height=total_height
                            #digit+= prediction_classification_marks(straightened_image, x, y, total_width, total_height,yolo_digit)
                            digit+= prediction_classification_marks(straightened_image, x_ocr, y_ocr,ocr_width,ocr_height,yolo_digit)
                        digit_ocr['Total_Marks_OCR'] = digit
                       
                digit_ocr_char=digit_ocr.get('Total_Marks_OCR')
                if '-' not in digit_ocr_char and '*' not in digit_ocr_char:
                    dict_ocr_value=int(digit_ocr.get('Total_Marks_OCR'))
                else:
                    dict_ocr_value=0   
                return dict_ocr_value
            

            def update_total_attempted_question(result_row):

                count_attempted_question=0
                for key,val in result_row.items():
                    if key.startswith('Q'):
                        if val !='00':
                            count_attempted_question=count_attempted_question+1
                result_row["Total_Attempted_Column"]=count_attempted_question
                return result_row

       
            '''recieves single or batch of images'''
         
            #@profile
            def condition1(result_row,batch_100_updated,q_value,total_final,label):# total sum equal
                    result_row['updation']=label
                    result_row['Flag']="P"
                    result_row['sum_question']=q_value
                    result_row['Total_Marks']=total_final
                    batch_100_updated.append(result_row)
                    return batch_100_updated
            
            def condition2(result_row,batch_100_updated,q_value,total_final,label):
                result_row['updation']=label
                result_row['Flag']="F"
                result_row['Total_Marks']=total_final
                result_row['sum_question']=q_value
                batch_100_updated.append(result_row)
                return batch_100_updated

            def processing_part_marks(batch_filenames,count,batch_count,image_batches):
                
                first_string = column_name_user[0]
                column_list11 = [item.strip() for item in first_string.split(',')] 
                batch_100_updated=[]
                for filename in batch_filenames:
                    path= os.path.join(folder_path, filename)
               
                    batch_straightened_images = straighten_images_with_yolo(
                        path,
                        SI, count, batch_count, image_batches, yolo_model_pd, omr_type_list1
                    )

                    batch_results = []
                    
                    for straightened_image in batch_straightened_images:
                        if straightened_image is not None :
                            result_row = process_image_marks(straightened_image,column_list11,filename)
                            result_row =update_total_attempted_question(result_row)
                            batch_results.append(result_row)
                   
                    for result_row in batch_results:
                        q_count=0
                        q_value=0 
                        total_final=result_row.get("Total_Marks")
                        for key in result_row:
                            if key.startswith('Q'):
                                q_count += 1
                                value = result_row.get(key)                               
                                # Convert the string value to a list of characters
                                value_chars = list(value)  # Use a different variable name for list conversion                              
                                blank_index_list = []
                                for i, x in enumerate(value_chars):
                                    if x == '-':
                                        blank_index_list.append(i)

                                # Case 1: If there are two dashes
                                if len(blank_index_list) == 2:
                                    for i in range(len(value_chars)):
                                        if value_chars[i] == '-':
                                            value_chars[i] = '0'  # Replace dash with 0

                                # Case 2: If there's one dash and it's at the start
                                elif len(blank_index_list) == 1 and blank_index_list[0] == 0:
                                    for i in range(len(value_chars)):
                                        if value_chars[i] == '-':
                                            value_chars[i] = '0'  # Replace dash with 0

                                # Case 3: If there's one dash and it's at index 1 (end)
                                elif len(blank_index_list) == 1 and blank_index_list[0] == 1:
                                    value_chars.pop(blank_index_list[0])  # Remove the dash
                                
                                value_chars = ''.join(value_chars)
                                if '*' not in value_chars:
                                    # Check if value_chars is not empty before converting to int
                                    if value_chars:
                                        value_chars = int(value_chars)
                                        q_value += value_chars
                                    else:
                                        print("Warning: value_chars is empty, skipping conversion.")
                        #print(" inital total_final",total_final)          
                        value_total_chars=list(total_final)
                        blank_index_list_total = []
                        
                        for i, x in enumerate(value_total_chars):  #### check - found at index 3 in total_marks
                            if x == '-':
                                blank_index_list_total.append(i)
                        if len(blank_index_list_total) == 1 and blank_index_list_total[0] == 2:
                                value_chars11=value_total_chars[:2]
                                total_final11=''.join(value_chars11)
                                total_final= total_final11
                        else:
                              total_final = total_final      
                        #print(" after changes total_final",total_final)
                        if '-' not in total_final and '*' not in total_final:
                           
                            total_final = int(total_final)
                            if total_final==q_value:
                                label="total_equal_sum"
                                batch_100_updated=condition1(result_row,batch_100_updated,q_value,total_final,label)
                                # result_row['updation']="total_equal_sum"
                                # result_row['Flag']="P"
                                # result_row['sum_question']=q_value
                                # result_row['Total_Marks']=total_final
                                # batch_100_updated.append(result_row)

                            else:
                                # digit ocr
                                dict_ocr_value=digit_ocr(batch_straightened_images) 
                                if dict_ocr_value==q_value:
                                        label="digit_ocr_equal_sum"
                                        batch_100_updated=condition1(result_row,batch_100_updated,q_value,dict_ocr_value,label)
                                        # result_row['Total_Marks']=dict_ocr_value
                                        # result_row['updation']="digit_ocr_equal_sum"
                                        # result_row['Flag']="P"
                                        # result_row['sum_question']=q_value
                                        # batch_100_updated.append(result_row)

                                elif dict_ocr_value==total_final:
                                    label="Teacher_Mistake(total_digitOcr_equal)"
                                    batch_100_updated=condition2(result_row,batch_100_updated,q_value,total_final,label)
                                    # result_row['updation']="Teacher_Mistake(total_digitOcr_equal)"
                                    # result_row['Flag']="F"
                                    # result_row['Total_Marks']=total_final
                                    # result_row['sum_question']=q_value
                                    # batch_100_updated.append(result_row)
                                   
                                else:
                                      ## bubble digit and sum not match  
                                    label="Wrong_prediction(Total_not_equal_sum)"
                                    batch_100_updated=condition2(result_row,batch_100_updated,q_value,total_final,label)   

                                    # result_row['updation']="Wrong_prediction(Total_not_equal_sum)"
                                    # result_row['Flag']="F"
                                    # result_row['Total_Marks']=total_final
                                    # result_row['sum_question']=q_value
                                    # batch_100_updated.append(result_row)
                       
                        else:
                           # this condition call when - found in Total Marks predicted by Bubble (marks blank)  
                            if len(blank_index_list_total) == 1 and blank_index_list_total[0] == 0:   
                                    
                                    value_total_list=list(total_final)
                                    value_chars12=value_total_list[1:]
                                    total_final12=''.join(value_chars12)
                                    total_final= total_final12
                                    total_final12=int(total_final12)
                                    if total_final12==q_value:
                                        label="total_equal_sum"
                                        batch_100_updated=condition1(result_row,batch_100_updated,str(q_value),total_final,label)   
                                        # result_row['Total_Marks']=total_final
                                        # result_row['updation']="total_equal_sum"
                                        # result_row['Flag']="P"
                                        # result_row['sum_question']=str(q_value)
                                        # batch_100_updated.append(result_row)
                                    else:
                                        label="Wrong_prediction(Total_not_equal_sum)"
                                        batch_100_updated=condition2(result_row,batch_100_updated,str(q_value),total_final,label)   


                                        # result_row['Total_Marks']=total_final
                                        # result_row['updation']="Wrong_prediction(Total_not_equal_sum)"
                                        # result_row['Flag']="F"
                                        # result_row['sum_question']=str(q_value)
                                        # batch_100_updated.append(result_row)


                            else:   
                       
                                    dict_ocr_value=digit_ocr(batch_straightened_images)  
                                    if dict_ocr_value==q_value:
                                        label="digit_ocr_equal_sum"
                                        batch_100_updated=condition1(result_row,batch_100_updated,str(q_value),dict_ocr_value,label)   

                                            # result_row['updation']="digit_ocr_equal_sum"
                                            # result_row['Flag']="P"
                                            # result_row["Total_Marks"]=dict_ocr_value
                                            # result_row['sum_question']=q_value
                                            # batch_100_updated.append(result_row)

                                    else:     
                                        label="Wrong_prediction(blank_double_found)"
                                        batch_100_updated=condition2(result_row,batch_100_updated,str(q_value),total_final,label)                             
                                        # result_row['updation']="Wrong_prediction(blank_double_found)"
                                        # result_row['Flag']="F"
                                        # result_row['sum_question']=q_value
                                        # batch_100_updated.append(result_row)

 
            
                del batch_straightened_images 
                return batch_100_updated 
  
###########################################pd part started here############################

            def check_pd_column_in_database(table_name, column_name,col_value):
                try:
                    # Establish a database connection
                    conn = psycopg2.connect(
                        host=host_name,
                        port=port_name,
                        database=db_name,
                        user=user_name,
                        password=password_db
                    )
                    # Create a cursor object to execute queries
                    cursor = conn.cursor()

                    # Build SQL query using proper placeholders
                    query = f"SELECT * FROM {table_name} WHERE {column_name} = %s"
                    
                    # Execute the parameterized query to prevent SQL injection
                    cursor.execute(query, (col_value,))
                    
                    # Fetch one record from the result
                    existing_user = cursor.fetchone()

                    # Check if the roll number exists
                    if existing_user:
                        return 1  # Roll number exists
                    else:
                        return 0  # Roll number does not exist

                except psycopg2.Error as e:
                    return 0, str(e)  # Return 0 and error message in case of failure

                finally:
                    # Ensure that cursor and connection are properly closed
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()
 
            def call_digit_model(straightened_image,yolo_model_pd_digit,col_value_orginal,digit_cordinates):
                print("inside call_digit_model")
                index = [i for i, letter in enumerate(col_value_orginal) if letter == '-' or letter == '*']
                col_value_original_list=list(col_value_orginal)# converting omr roll number into list
                for i in index:
                    points=digit_cordinates[i]
                    x, y, width,height = points
                    roll_classification = prediction_classification_pd(straightened_image, x, y,width,height,yolo_model_pd_digit)
                    col_value_original_list[i]=roll_classification 
                # print("roll_classification",roll_classification)
                print("omr_roll_no original",col_value_orginal)
                col_value_original_list_str = ''.join(col_value_original_list)### converting omr roll number after classification in string
                col_value_original_list_str_length=len(col_value_original_list_str)
                print("omr_roll_no_str ocr",col_value_original_list_str )
                return col_value_original_list_str  ,col_value_original_list_str_length
 
            def process_damage(straightened_image,col_value_orginal,ocr_column_name,dublicate,result_row,ocr_row ,
                    digit_cordinate,Prediction_Type,majority_length ):
                print("inside process damage")
                col_value_str ,col_value_str_length= call_digit_model(straightened_image,yolo_model_pd_digit,col_value_orginal,digit_cordinate) 
                print(" original value",col_value_orginal)
                print(" value after ocr",col_value_str)
                if pd.notnull(col_value_str ) and "-" not in col_value_str  and "*" not in col_value_str  and col_value_str_length==majority_length:
                    if col_value_str not in dublicate:### check duplicate(dublicate not found)                    
                        dublicate.append(col_value_str) 
                        roll_return_admit_card_db=check_pd_column_in_database(admit_card_table_name,ocr_column_name,col_value_str)
                        if roll_return_admit_card_db==1:# roll found in admit card table
                            result_row[ocr_column_name]=col_value_str
                            ocr_row[Prediction_Type]="ocr"
                            print("roll found in database")
                        else: # roll not found in admit card table
                            print("roll not found in database")
                            ocr_row[ocr_column_name]=col_value_orginal
                            ocr_row[Prediction_Type]="original"
                    else:# dublicate found
                        print("dublicate found")
                        ocr_row[ocr_column_name]=col_value_orginal
                        ocr_row[Prediction_Type]="original"
                    
                else: # *, _ and length of digit prediction found
                    ocr_row[ocr_column_name]=col_value_orginal
                    ocr_row[Prediction_Type]="original"

            def find_damage_col(col_value,col_value_length,majority_length):
                if pd.notnull(col_value) and "-" not in col_value  and "*" not in col_value  and col_value_length==majority_length :
                    #database_omr_insert_table(db_file,correct_table_name,result_row,conn_admit_card)
                    return 0 # no damage    
                else:
                    return 1 # damage 

            def process_pd_ocr(straightened_image,result_row,pd_column_name_admit_card):

                first_string = pd_column_name_admit_card[0]
                pd_ocr_list= [item.strip() for item in first_string.split(',')]
                
                file_name = result_row["Filename"]
                print("filename",file_name)
                ocr_row = {'Filename': os.path.basename(file_name)}

                if "Ansidrno" in pd_ocr_list:
                    print("Ansidrno")
                    global Ansidrno_dublicate
                    majority_length=len(Ansidrno_digit_cordinates)
                    Ansidrno_dublicate=[]
                    ocr_column_name="Ansidrno" 
                    Prediction_Type=  "Ansidrno_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Ansidrno_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Ansidrno_dublicate,result_row,ocr_row,
                                        Ansidrno_digit_cordinates,Prediction_Type,majority_length)
                
                if "Center_Code" in pd_ocr_list:
                    print(" Center_Code_OCR ")
                    global Center_dublicate
                    majority_length=len(Center_Code_digit_cordinates)
                    Center_dublicate=[]
                    ocr_column_name="Center_Code" 
                    Prediction_Type=  "Center_Code_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Center_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Center_dublicate,result_row,ocr_row,
                                        Center_Code_digit_cordinates,Prediction_Type,majority_length)

                if "Paper_No" in pd_ocr_list:
                    print(" Paper_No_OCR ")

                    majority_length=len(Paper_No_digit_coordinates)
                    Center_dublicate=[]
                    ocr_column_name="Paper_No" 
                    Prediction_Type=  "Paper_No_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Center_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Center_dublicate,result_row,ocr_row,
                                        Paper_No_digit_coordinates,Prediction_Type,majority_length)
                    
                if "College_Code" in pd_ocr_list:
                    print(" College_Code_OCR ")
                    
                    majority_length=len(College_Code_digit_cordinates)
                    Center_dublicate=[]
                    ocr_column_name="College_Code" 
                    Prediction_Type=  "College_Code_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Center_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Center_dublicate,result_row,ocr_row,
                                        College_Code_digit_cordinates,Prediction_Type,majority_length)
          


                if "Date_of_Exam" in pd_ocr_list:
                    print("Date_of_Exam ocr")
                    global Date_of_Exam_dublicate
                    majority_length=len(Exam_digit_cordinates)
                    Date_of_Exam_dublicate=[]
                    ocr_column_name="Date_of_Exam" 
                    Prediction_Type=  "Date_of_Exam_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Date_of_Exam_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Date_of_Exam_dublicate,result_row,ocr_row,
                                        Exam_digit_cordinates,Prediction_Type,majority_length)

                if "Paper_Code" in pd_ocr_list:
                    print("Paper_Code ocr")
                    global Paper_Code_dublicate
                    majority_length=len(Paper_digit_cordinates)
                    Paper_Code_dublicate=[]
                    ocr_column_name="Paper_Code" 
                    Prediction_Type=  "Paper_Code_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Paper_Code_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Paper_Code_dublicate,result_row,ocr_row,
                                        Paper_digit_cordinates,Prediction_Type,majority_length)
                        

                
                if "Subject_Code" in pd_ocr_list:
                    print("Subject_Code ocr")
                    global Subject_Code_dublicate
                    majority_length=len(Subject_Code_OCR_cordinates)
                    Subject_Code_dublicate=[]
                    ocr_column_name="Subject_Code" 
                    Prediction_Type=  "Subject_Code_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Paper_Code_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Subject_Code_dublicate,result_row,ocr_row,
                                        Subject_Code_OCR_cordinates,Prediction_Type,majority_length)
                              
                # return result_row,ocr_row
            
                if "Paper_Code_2" in pd_ocr_list:
                    print("Paper_Code_2 ocr")
                    global Paper_Code_2_dublicate
                    majority_length=len(Paper_digit_cordinates)
                    Paper_Code_2_dublicate=[]
                    ocr_column_name="Paper_Code" 
                    Prediction_Type=  "Paper_Code_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace

                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Paper_Code_2_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Paper_Code_dublicate,Paper_Code_2_dublicate,result_row,ocr_row,
                                        Paper_digit_cordinates,Prediction_Type,majority_length)
                              
                return result_row,ocr_row

# --------- Have changed this function to insert 50 data at once in table by ashutosh ------------------------------------     
            '''Data comes in batches and processed one by one and inserted the data in the postgres database for PD. internally it 
            calls two other functions process_image and process_pd_ocr which will be called if user has given the admit card table name.'''
            
            #@profile
            def processing_part_pd(batch_filenames,count,batch_count,image_batches):
            
                first_string = column_name_user[0]
                pd_ocr_list= [item.strip() for item in first_string.split(',')]
                pd_ocr_list_prediction=[f'{x}_Prediction_Type' for x in pd_ocr_list]
                batch_of_results=[]
                batch_pd_ocr_list=[]
                for filename in batch_filenames:
                    path= os.path.join(folder_path, filename)

                    batch_straightened_images = straighten_images_with_yolo(
                        path,
                        SI, count, batch_count, image_batches, yolo_model_pd, omr_type_list1
                    )

                    batch_results = []
            
                    for straightened_image in batch_straightened_images:
                        if straightened_image is not None :
                            result_row = process_image_pd(straightened_image,pd_ocr_list,filename)
                            batch_results.append(result_row)
                    
              
                    for result_row in batch_results:
                        if admit_card_available==1:
                            pd_ocr_list_prediction=[f'{x}_Prediction_Type' for x in pd_ocr_list]  #### creating column name with suffix Prediction_Type
                            result_row_response,ocr_row_response=process_pd_ocr(straightened_image,result_row,pd_column_name_admit_card)
                            # database_process(table_name,result_row_response,pd_ocr_list)
                            database_process(f'{table_name}_Rechecked',ocr_row_response,pd_ocr_list_prediction)
                            batch_of_results.append((result_row_response,pd_ocr_list))   
                        else:
                            batch_of_results.append((result_row,pd_ocr_list))
                del batch_straightened_images  
                return batch_of_results


            def delete_files_in_folder_recursive(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        os.remove(file_path)

###########################################################################################################
            ''' process_images_with_batches Overview
                This function processes a batch of images based on user-defined parameters, such as batch size and a specific range of images. It is designed to:

                Handle batch processing when no range is specified.
                Handle individual image processing when a range is defined.
                Calculate and display progress, estimated time remaining, and other status updates during processing.
                Save the processed data to a database in chunks to optimize performance.'''
        
            #@profile
            def process_images_with_batches(folder_path, batch_size, omr_type_list1):
                global total_image_length,total_range
          
                count = 0
                start_time=0.0
                image_filenames = [filename for filename in os.listdir(folder_path) if filename.lower().endswith((".jpg", ".jpeg", ".png"))]
                image_batches = [image_filenames[i:i + batch_size] for i in range(0, len(image_filenames), batch_size)]
                total_image_length = len(image_batches)
                total_range=0
                file_range_count = 0
                #range_count = 0
                image_batches1 = [[os.path.splitext(filename)[0] for filename in batch] for batch in image_batches]

                filename_start_number = str(file_start_range)
                filename_end_number = str(file_end_range)

                if file_end_range!=0 :
                    start_index = find_index(image_batches1, filename_start_number)
                    end_index = find_index(image_batches1, filename_end_number)
  
                    if start_index != -1 or end_index != -1:
                        #print("Start or end index not found in the list")
                        total_range = end_index - start_index
                        total_range = total_range +1
                        image_batches=image_batches[start_index :end_index+1]
                
              
                try:
                    batch_of_100=[]
                    batch_of_100_pd_1=[]
                    batch_of_100_pd_2=[]
                    first_string = column_name_user[0]
                    column_list11 = [item.strip() for item in first_string.split(',')] 
                    flag=0
                    
                    ''' -------This code processes batches of images, estimates processing times, and stores the 
                    results in a database based on the type of operation specified in the omr_type_list1. '''

                    for batch_count, batch_filenames in enumerate(image_batches, start=1):
                        
                        if stop == 1:
                            QMessageBox.showwarning("Warning", "Close button is pressed. Terminating batch processing.")
                            break

                        if file_end_range==0:
                            total_image_length = len(image_batches)### for batch_processing total image display
                            self.update_progress.emit(batch_count)
                            
                            total_image_left= total_image_length-batch_count
                            #print("total_image_left",total_image_left)

                            if count==10:
                                end_time = time.time()
                                total_time_taken=end_time-start_time
                                #total_time_taken=total_time_taken/60 #### minutes
                                
                                time_taken_one_image=total_time_taken/10 ##### one image
                                #self.timer_one_image.emit(time_taken_one_image)
                                #total_Processing_time_all_image=time_taken_one_image*(total_image_length-10)
                                total_Processing_time_all_image=time_taken_one_image*(total_image_left)
                    
                            if count>10:
                                total_Processing_time_all_image=total_Processing_time_all_image-time_taken_one_image
                                remaining_time=total_Processing_time_all_image
                                hours, remainder = divmod(remaining_time, 3600)
                                minutes, seconds = divmod(remainder, 60)
                                if is_scientific_notation(str(remaining_time)):

                                    time_left=f'0 hr 0 min 0 sec'
                                    self.omr_remaining_time.emit(time_left)
                                else:
                                    time_left=f'{int(hours)} hr {int(minutes)} min {int(seconds)} sec'
                                    self.omr_remaining_time.emit(time_left)
                                
                                if count==250:
                                    count=0
                                    #print("count value is zero",count)
                                    time_left=f'wait.......'
                                    self.omr_remaining_time.emit(time_left)
                            
                            if count==0:
                                start_time=time.time()
                                
                            count=count_value(count)
                            if 'PD'in  omr_type_list1:  
                                #data=processing_part_pd(batch_filenames,count,batch_count,image_batches)
                                try:
                                    data=processing_part_pd(batch_filenames,count,batch_count,image_batches)
                                    batch_of_100_pd_1.append(data)   
                                    if  (len(batch_of_100_pd_1)*batch_size)>=100:
                                        database_process3(f"{table_name}",batch_of_100_pd_1)
                                        batch_of_100_pd_1=[]
                                        delete_files_in_folder_recursive(straightened_folder)
                                except:
                                    print("Some Image is not been processed 2574")
                                    continue

                            if 'Marks_Anchor_Top'in  omr_type_list1 :  
                                #  processing_part_marks(batch_filenames,count,batch_count,image_batches)
                                batch_data=processing_part_marks(batch_filenames,count,batch_count,image_batches)
                                batch_of_100.append(batch_data)
                                if (len(batch_of_100)*batch_size)>=100:
                                    database_process2(f"{table_name}",batch_of_100,column_list11)
                                  
                                    batch_of_100=[]
                                    delete_files_in_folder_recursive(straightened_folder)

                            if 'Marks_Anchor_Bottom'in  omr_type_list1: 
                                batch_data=processing_part_marks(batch_filenames,count,batch_count,image_batches)
                                batch_of_100.append(batch_data)
                                if  (len(batch_of_100)*batch_size)>=100:
                                    database_process2(f"{table_name}",batch_of_100,column_list11)
                                    
                                    batch_of_100=[]
                                    delete_files_in_folder_recursive(straightened_folder)

                                  
                        else:
                            '''Processes only one file at a time within the specified range.'''  
                            total_image_length = total_range ### for batch_processing total image display
                            filename_start_number =str(file_start_range)
                            filename_end_number = str(file_end_range)
                            
                            for filename in batch_filenames:
                                file_split=os.path.splitext(os.path.basename(filename))[0]
                                Scanno_range=file_split    
                                if Scanno_range >= filename_start_number:
                                
                                    if Scanno_range > filename_end_number:
                                        #print(f"Stopping execution as Scanno_range {Scanno_range} is greater than filename_end_number {filename_end_number}")
                                        flag=1

                                    file_range_count=file_range_count+1 
                                    if file_range_count < total_range+1:
                                        self.update_progress.emit(file_range_count) 
                                    #################################################################################
                                        total_image_left= total_image_length-file_range_count
                                        if count==10:
                                            end_time = time.time()
                                            total_time_taken=end_time-start_time    
                                            time_taken_one_image=total_time_taken/10 ##### one image
                                            total_Processing_time_all_image=time_taken_one_image*(total_image_left)
                                
                                        if count>10:
                                            total_Processing_time_all_image=total_Processing_time_all_image-time_taken_one_image
                                            remaining_time=total_Processing_time_all_image
                                            hours, remainder = divmod(remaining_time, 3600)
                                            minutes, seconds = divmod(remainder, 60)
                                            if is_scientific_notation(str(remaining_time)):

                                                time_left=f'0 hr 0 min 0 sec'
                                                self.omr_remaining_time.emit(time_left)
                                            else:
                                                time_left=f'{int(hours)} hr {int(minutes)} min {int(seconds)} sec'
                                                self.omr_remaining_time.emit(time_left)
                                            
                                            if count==250:
                                                count=0
                                                #print("count value is zero",count)
                                                time_left=f'wait.......'
                                                self.omr_remaining_time.emit(time_left)
                                        

                                        
                                        if count==0:
                                            start_time=time.time()
                                            
                                        count=count_value(count)
                                        if 'PD'in  omr_type_list1:  
                              
                                            data=processing_part_pd(batch_filenames,count,batch_count,image_batches)
                                            batch_of_100_pd_1.append(data)
                                            if  (len(batch_of_100_pd_1)*batch_size)>=100:
                                                database_process3(f"{table_name}",batch_of_100_pd_1)
                                                batch_of_100_pd_1=[]
                                                delete_files_in_folder_recursive(straightened_folder)
                     

                                            # try:
                                            #     data=processing_part_pd(batch_filenames,count,batch_count,image_batches)
                                            #     batch_of_100_pd_1.append(data)
                                            #     if  (len(batch_of_100_pd_1)*batch_size)>=100:
                                            #         database_process3(f"{table_name}",batch_of_100_pd_1)
                                            #         batch_of_100_pd_1=[]
                                            #         delete_files_in_folder_recursive(straightened_folder)
                                            # except:
                                            #     print("Some Image is not been processed  26666")
                                            #     continue

                                        if 'Marks_Anchor_Top'in  omr_type_list1 :
                                              
                                            batch_data=processing_part_marks(batch_filenames,count,batch_count,image_batches)
                                            batch_of_100.append(batch_data)
                                            if (len(batch_of_100)*batch_size)>=100:
                                                
                                              
                                                database_process2(f"{table_name}",batch_of_100,column_list11)
                                                batch_of_100=[]
                                                delete_files_in_folder_recursive(straightened_folder)
                                        
                                        if 'Marks_Anchor_Bottom'in  omr_type_list1: 
                                            batch_data=processing_part_marks(batch_filenames,count,batch_count,image_batches)
                                            batch_of_100.append(batch_data)
                                            if  (len(batch_of_100)*batch_size)>=100:
                                                database_process2(f"{table_name}",batch_of_100,column_list11)
                                              
                                                batch_of_100=[]
                                                delete_files_in_folder_recursive(straightened_folder)
                            

                        if flag==1:
                            break

                    
                    if len(batch_of_100)>0:
                        database_process2(f"{table_name}",batch_of_100,column_list11)
                        batch_of_100=[]
                    
                    else:
                        database_process3(f"{table_name}",batch_of_100_pd_1)
                        batch_of_100_pd_1=[]

                    # if ('Marks_Anchor_Top' in  omr_type_list1) | ('Marks_Anchor_Bottom'in  omr_type_list1) :
                    #     pass
                     
                    gc.collect()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error during batch processing: {e}")
                # except ValueError as ve:
                #      logging.exception("Exception occurred: %s", str(ve))
 
################################################################################################################################

            '''Calling the function '''
            

            process_images_with_batches(folder_path, batch_size, omr_type_list1)
            #print("filename_done_by_segmentation",filename_done_by_segmentation)

            if os.path.exists(straightened_folder):
                try:
                    shutil.rmtree(straightened_folder)
                except Exception as e:
                    print(f"There has been an error:  {e}")     




class ScriptLauncher(QMainWindow):
   
    def __init__(self):

        super().__init__()
        self.setWindowTitle("OMR")
        self.setGeometry(0, 0, 400, 100)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        name_label = QLabel("OMR SHARP (PD+Marks)", self)
        name_label.setStyleSheet("color: lightyellow; background-color: purple;")
        name_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        font = name_label.font()
        font.setPointSize(15)
        font.setBold(True)
        name_label.setFont(font)
        layout.addWidget(name_label)
        self.input_folder_path_label, input_folder_button,self.input_folder_path= self.add_input_folder(layout,"Input Folder-", "Browse", "Enter Image Folder Path")
        input_folder_button.setFixedSize(200, 20)
        self.input_folder_path.setFixedSize(400, 20)
        input_folder_button.clicked.connect(self.browse_input_folder1)        
        self.template_label, template_button, self.template_path = self.add_temp_folder(layout,"Reference Image for finding similarity -", "Browse","Enter template path or use a default reference")
        template_button.setFixedSize(200, 20)
        self.template_path.setFixedSize(400, 20)
        template_button.clicked.connect(self.browse_template_path1) 
        
        dimension_layout = QHBoxLayout()
        self.Image_Size_label=QLabel("Image Size:")
        self.image_Width_label = QLabel("Width", self)
        self.image_Width = QLineEdit(self)
        self.image_Width.setText("1654")

        self.image_Height_label = QLabel("Height", self)
        self.image_Height = QLineEdit(self)
        self.image_Height.setText("2366")

        self.Image_Size_button = QPushButton('Image Size')
        self.Image_Size_button .setStyleSheet("background-color: darkCyan; color: white;")
        self.Image_Size_button.clicked.connect(self.find_image_size)
        
        dimension_layout.addWidget(self.Image_Size_label )
        dimension_layout.addWidget(self.image_Width_label)
        dimension_layout.addWidget(self.image_Width)
        dimension_layout.addWidget(self.image_Height_label)
        dimension_layout.addWidget(self.image_Height)
        dimension_layout.addWidget(self.Image_Size_button)
        layout.addLayout(dimension_layout)

        file_range = QHBoxLayout()
        self.image_start_label = QLabel("File Name Start Index -", self)
        self.image_start = QLineEdit(self)
        self.image_start.setPlaceholderText("Enter Start Index")
        self.image_start.setFixedSize(100, 20)
        self.image_start.setText("-")

        self.image_end_label = QLabel("File Name End Index -", self)
        self.image_end = QLineEdit(self)
        self.image_end.setPlaceholderText("Enter End Index")
        self.image_end.setFixedSize(100, 20)
        self.image_end.setText("-")

        self.b1 = QCheckBox("All")
        self.b1.setChecked(False)  # Set the checkbox to default to checked
        self.b1.stateChanged.connect(lambda: self.btnstate(self.b1))
         
        file_range.addWidget(self.image_start_label)
        file_range.addWidget(self.image_start)
        file_range.addWidget(self.image_end_label)
        file_range.addWidget(self.image_end)
        file_range.addWidget(self.b1)
        layout.addLayout(file_range)
       
        self.list_widget_question = QListWidget()
        self.list_widget_question.addItems(["PD","Marks_Anchor_Top","Marks_Anchor_Bottom"])
        self.list_widget_question.setSelectionMode(QListWidget.MultiSelection)
        self.list_widget_question.itemSelectionChanged.connect(self.update_question_options)
        layout.addWidget(self.list_widget_question)
        self.columns_platform_question = QComboBox()
        layout.addWidget(self.columns_platform_question)
       
        straight_image_layout = QHBoxLayout()
        self.pbar = QProgressBar(self)
        self.pbar.setFixedSize(600, 20)
        self.pbar.setRange(0, 10)
        straight_image_layout.addWidget(self.pbar)
        self.Processing = QLineEdit(self)
        straight_image_layout.addWidget(self.Processing)
        self.submitbtn1 = QPushButton(" straight image ", self)
        self.submitbtn1.clicked.connect(self.process_straight_one_image)
        self.submitbtn1.setStyleSheet("background-color: darkCyan; color: white;")
        straight_image_layout.addWidget(self.submitbtn1)
        layout.addLayout(straight_image_layout)

        self.template_label2, template_button2, self.template_path2 = self.add_input_group_template(layout, "Reference Image  -", "Browse", "Enter template path or use a default reference")
        template_button2.setFixedSize(200, 20)
        self.template_path2.setFixedSize(400, 20)
        template_button2 .clicked.connect(self.browse_template_path2)             

        self.template_label1 = QLabel("Cordinates points selection -", self)
        layout.addWidget(self.template_label1)
        self.submitbtn = QPushButton("Select Co-Ordinates", self)
        self.submitbtn.clicked.connect(self.threaded_open_html_page)
        self.submitbtn.setStyleSheet("background-color: darkCyan; color: white;")
        layout.addWidget(self.submitbtn)

        self.cordinates_label, cordinate_button, self.cordinate_path = self.add_input_group_coordinates(layout, "Cordinates File -", "Browse", "Upload Cordinates File")
        cordinate_button.setFixedSize(200, 20)
        self.cordinate_path.setFixedSize(400, 20)
        cordinate_button.clicked.connect(self.browse_cordinate_path)
        self.Image_Size_button1 = QPushButton('OMR Pattern ')
        self.Image_Size_button1 .setStyleSheet("background-color: darkCyan; color: white;")
        self.Image_Size_button1.clicked.connect(self.cordinates_point_text_file)
        layout.addWidget(self.Image_Size_button1)
        
        omr = QHBoxLayout()
        self.omr_pattern= QLabel(" OMR Details -", self)
        omr.addWidget(self.omr_pattern) 
        self.columns_platform =QLineEdit() #QComboBox()
        omr.addWidget(self.columns_platform) 
        layout.addLayout(omr) 

        ocr_type = QHBoxLayout()
        self.ocr_question= QLabel(" OCR Details ", self)
        ocr_type.addWidget(self.ocr_question) 
        self.columns_platform_ocr =QLineEdit() # QComboBox()
        ocr_type.addWidget(self.columns_platform_ocr)
        layout.addLayout(ocr_type)
        db_layout = QHBoxLayout()

        self.similarity_per_label = QLabel("SI Index", self)
        db_layout.addWidget(self.similarity_per_label)
        self.similarity_per = QLineEdit(self)
        self.similarity_per.setPlaceholderText("Enter similarity percentage")
        self.similarity_per.setText("0")  # Set default value
        db_layout.addWidget(self.similarity_per)

        self.dbname_label = QLabel('DB Name:')
        db_layout.addWidget(self.dbname_label)
        self.dbname_lineedit = QLineEdit()
        db_layout.addWidget(self.dbname_lineedit)

        self.user_label = QLabel('User:')
        db_layout.addWidget(self.user_label)
        self.user_lineedit = QLineEdit()
        db_layout.addWidget(self.user_lineedit)

        self.password_label = QLabel('Password:')
        db_layout.addWidget(self.password_label)
        self.password_lineedit = QLineEdit()
        self.password_lineedit.setEchoMode(QLineEdit.Password)
        db_layout.addWidget(self.password_lineedit)

        layout.addLayout(db_layout)

        db_layout1 = QHBoxLayout()

        self.host_label = QLabel('Host:')
        db_layout1.addWidget(self.host_label)
        self.host_lineedit = QLineEdit()
        db_layout1.addWidget(self.host_lineedit)

        self.port_label = QLabel('Port:')
        db_layout1.addWidget(self.port_label)
        self.port_lineedit = QLineEdit()
        db_layout1.addWidget(self.port_lineedit)

        self.table_label = QLabel("OMR(Table Name):", self)
        db_layout1.addWidget(self.table_label)
        self.table_value = QLineEdit(self)
        db_layout1.addWidget(self.table_value)
        layout.addLayout(db_layout1)
        
        self.admit_card_table_label = QLabel("Admit_Card(Table Name):", self)
        db_layout1.addWidget(self.admit_card_table_label)
        self.admit_card_table_value = QLineEdit(self)
        db_layout1.addWidget(self.admit_card_table_value)

        layout.addLayout(db_layout1)
        
        self.load_saved_db_config()

        self.progress_label = QLabel("Batch Processing Result", self)
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.BatchProcessing_label_path = QLabel(self)
        layout.addWidget(self.BatchProcessing_label_path)
        time_layout = QHBoxLayout()


        self.All_image_time_label = QLabel("  Estimated  Time -", self)
        self.All_image_time_value = QLineEdit(self)
        self.All_image_time_value.setText("0:0")  # Set default value
        #self.one_image_time_value.setFixedSize(200, 20)
        self.All_image_time_label .setStyleSheet("color:blue")
        font = self.All_image_time_label .font()
        font.setPointSize(10)
        font.setBold(True)
        self.All_image_time_label.setFont(font)

        self.All_image_time_value .setStyleSheet("color:red")
        font = self.All_image_time_value.font()
        font.setPointSize(10)
        font.setBold(True)
        self.All_image_time_value.setFont(font)

        self.end_time_label = QLabel(" Total Time Taken -", self)
        self.end_time_value = QLineEdit(self)
        self.end_time_value.setText("0:0") 
        #self.end_time_value.setFixedSize(200, 20)
        self.end_time_label.setStyleSheet("color:blue")
        font = self.end_time_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.end_time_label.setFont(font)
        self.end_time_value.setStyleSheet("color:red")
        font = self.end_time_value.font()
        font.setPointSize(10)
        font.setBold(True)
        self.end_time_value.setFont(font)


        time_layout.addWidget(self.All_image_time_label)
        time_layout.addWidget(self.All_image_time_value)
        time_layout.addWidget(self.end_time_label)
        time_layout.addWidget(self.end_time_value)
        layout.addLayout(time_layout)


        self.submitbtn = QPushButton("Start", self)
        self.submitbtn.setFixedSize(200, 30)
        layout.addStretch()
        self.submitbtn.clicked.connect(self.gui_variable)  
        self.submitbtn.setStyleSheet("background-color: green; color: white;")

        closebtn = QPushButton("Close", self)
        closebtn.setFixedSize(200, 30)
        closebtn.clicked.connect(self.close_window)
        closebtn.setStyleSheet("background-color: red; color: white;")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.submitbtn)
      

        button_layout.addWidget(closebtn)

        layout.addLayout(button_layout)

    

    # def gui_variable(self):
    #     # Retrieve values from GUI inputs
    
    #     time_left_new=f'0 hr 0 min 0 sec'
    #     total_time_hours_new=f'0 hr 0 min 0 sec'
    #     self.All_image_time_value.setText(f'{time_left_new}')        
    #     self.end_time_value.setText(f'{total_time_hours_new}')

    #     print("####################ocr",self.columns_platform_ocr.text())
    #     self.table_name = self.table_value.text()
    #     self.admit_card_table_name=self.admit_card_table_value.text()
    #     self.db_name = self.dbname_lineedit.text()
    #     self.user_name = self.user_lineedit.text()
    #     self.password_db = self.password_lineedit.text()
    #     self.host_name = self.host_lineedit.text()
    #     self.port_name = self.port_lineedit.text()

    #     self.template_path=self.template_path2.text()
    #     self.folder_path = self.input_folder_path.text()
       
    #     self.cordinates_point_path = self.cordinate_path.text()

    #     file_start_value=self.image_start.text()
    #     file_end_value=self.image_end.text()
    #     omr_type= self.columns_platform_question.currentText()
    #     first_string = omr_type
    #     omr_type_list = [item.strip() for item in first_string.split(',')]
    #     pd_type=omr_type_list[0]
    #     print("pd_type",pd_type)
    #     print("length of omr_type_list ",len(omr_type_list))

    #     try:
    #         # Check if all required fields are filled
    #         if all([self.table_name,self.db_name, self.user_name, self.password_db, self.host_name, self.port_name,self.template_path2,self.folder_path,self.cordinates_point_path,pd_type ]):
    #             if check_saved_folder=='False' and file_start_value=='-' and file_end_value=='-':
    #                 #print("inside gui to check Check Box")
    #                 QMessageBox.information(self, "Input Error", "Please Click on  Check Box named as 'All' when File Name Start and End Index is '-'.")
    #                 self.refresh()
    #             else:
    #                 self.save_db_config()
    #                 self.thread_calling()
    #         else:

    #             QMessageBox.information(self, "Input Error", "Please fill in all required fields.")
    #     except Exception as e:
    #         QMessageBox.information(self, "Error", f"Database Entry Error: {e}")
    #         self.refresh()

    def gui_variable(self):
        # Retrieve values from GUI inputs
        
        time_left_new = f'0 hr 0 min 0 sec'
        total_time_hours_new = f'0 hr 0 min 0 sec'
        self.All_image_time_value.setText(f'{time_left_new}')        
        self.end_time_value.setText(f'{total_time_hours_new}')

        print("####################ocr", self.columns_platform_ocr.text())
        self.table_name = self.table_value.text()
        self.admit_card_table_name = self.admit_card_table_value.text()
        self.db_name = self.dbname_lineedit.text()
        self.user_name = self.user_lineedit.text()
        self.password_db = self.password_lineedit.text()
        self.host_name = self.host_lineedit.text()
        self.port_name = self.port_lineedit.text()

        self.template_path = self.template_path2.text()
        self.folder_path = self.input_folder_path.text()

        self.cordinates_point_path = self.cordinate_path.text()

        file_start_value = self.image_start.text()
        file_end_value = self.image_end.text()
        omr_type = self.columns_platform_question.currentText()
        first_string = omr_type
        omr_type_list = [item.strip() for item in first_string.split(',')]
        pd_type = omr_type_list[0]

        # List of required fields
        required_fields = {
            "Table Name": self.table_name,
            "Database Name": self.db_name,
            "User Name": self.user_name,
            "Password": self.password_db,
            "Host Name": self.host_name,
            "Port Name": self.port_name,
            "Template Path": self.template_path,
            "Folder Path": self.folder_path,
            "Cordinate Path": self.cordinates_point_path,
            "OMR Type": pd_type
        }

        # Check for missing fields
        missing_fields = [field for field, value in required_fields.items() if not value]

        try:
            # Check if all required fields are filled
            if missing_fields:
                missing_fields_str = ", ".join(missing_fields)
                QMessageBox.information(self, "Input Error", f"Please fill in the following fields: {missing_fields_str}")
            elif check_saved_folder == 'False' and file_start_value == '-' and file_end_value == '-':
                QMessageBox.information(self, "Input Error", "Please Click on the Check Box named as 'All' when File Name Start and End Index is '-'.")
                self.refresh()
            else:
                self.save_db_config()
                self.thread_calling()

        except Exception as e:
            QMessageBox.information(self, "Error", f"Database Entry Error: {e}")
            self.refresh()


    def save_db_config(self):
        config = {
            'dbname': self.db_name,
            'user': self.user_name,
            'password': self.password_db,
            'host': self.host_name,
            'port': self.port_name
        }
        config_path = os.path.join("C:\\postgres_config_path", 'db_config_pd_marks.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f)


    def find_image_size(self):
        folder_path = self.input_folder_path.text()
        filename_list = os.listdir(folder_path)        
        # Filter out non-image files
        image_files = [f for f in filename_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if not image_files:
            print("No image files found in the directory.")
            return
        
        # Select the first valid image
        filename = image_files[0]     
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        # Check if the image is loaded properly
        if image is None:
            print("Failed to load the image.")
            return
        
        height, width, _ = image.shape        
        self.image_Width.setText(f'{width}')
        self.image_Height.setText(f'{height}')

    def update_question_options(self):
        selected_items = [item.text() for item in self.list_widget_question.selectedItems()]
        self.columns_platform_question.clear()
        if selected_items:
            combined_text = ' , '.join(selected_items)
            
            if selected_items[0] == "Marks_Anchor_Top" or selected_items[0] == "Marks_Anchor_Bottom":
                QMessageBox.information(self, "Message", "Selecting a pattern like Total_Marks and Total_Marks_OCR along with the Question is mandatory" )  
            if selected_items[0] == "PD" :
                QMessageBox.information(self, "Message", "if you are uploading admit card then please select OCR option for each and every pattern")
            self.columns_platform_question.addItem(combined_text) 

    def add_input_folder(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()
        label = QLabel(label_text, self)
        input_layout.addWidget(label)
        button = QPushButton(button_text, self)
        input_layout.addWidget(button)
        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)
        layout.addLayout(input_layout)
        return label, button, line_edit

    def add_input_group_template(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()
        label = QLabel(label_text, self)
        input_layout.addWidget(label)
        button = QPushButton(button_text, self)
        input_layout.addWidget(button)
        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)
        layout.addLayout(input_layout)
        return label, button, line_edit

    def add_temp_folder(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()
        label = QLabel(label_text, self)
        input_layout.addWidget(label)
        button = QPushButton(button_text, self)
        input_layout.addWidget(button)
        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)
        layout.addLayout(input_layout)
        return label, button, line_edit 
    

    def add_similarity(self, layout, label_text, default_value):
        input_layout = QHBoxLayout()  # Create a QHBoxLayout for grouping the label and line edit
        label = QLabel(label_text, self)
        input_layout.addWidget(label)
        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(default_value)
        input_layout.addWidget(line_edit)
        layout.addLayout(input_layout)  # Add the QHBoxLayout to the main layout
        return label, line_edit
    
    def add_Table(self, layout, label_text, default_value):
        input_layout = QHBoxLayout()  # Create a QHBoxLayout for grouping the label and line edit
        label = QLabel(label_text, self)
        input_layout.addWidget(label)
        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(default_value)
        input_layout.addWidget(line_edit)
        layout.addLayout(input_layout)  # Add the QHBoxLayout to the main layout
        return label, line_edit

   
    def add_input_group_coordinates(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()
        label = QLabel(label_text, self)
        input_layout.addWidget(label)
        button = QPushButton(button_text, self)
        input_layout.addWidget(button)
        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)
        layout.addLayout(input_layout)
        return label, button, line_edit
  
    def thread_calling(self):
        self.submitbtn.setDisabled(True)
        self.worker_thread = WorkerThread(self)
        self.worker_thread.start()
        #self.worker_thread.timer_one_image.connect(self.omr_one_image_time)
        self.worker_thread.omr_remaining_time.connect(self.omr_remaining_time)
        self.worker_thread.end_timer.connect(self.omr_end_timer)
        self.worker_thread.finished.connect(self.result_done)
        self.worker_thread.update_progress.connect(self.batch_progress)
        self.worker_thread.db_error.connect(self.db_entry_error)
    

    def omr_remaining_time(self,time_left):
        self.All_image_time_value.setText(f'{time_left}')

    def omr_end_timer(self,total_time_hours):
        self.end_time_value.setText(f'{total_time_hours}')

    def result_done(self,error_result):
        if error_result==0:
            QMessageBox.information(self, "Result", "OMR Result Prepared")
        self.refresh_omr()
        self.submitbtn.setDisabled(False)

    def db_entry_error(self,db_connection_error):    
        QMessageBox.information(self, "Error", f"Database Connection Error: {db_connection_error}" )
        self.refresh_omr()

    def refresh(self):
        global error_result
        error_result=0

    def refresh_omr(self):
        global total_image_length
        total_image_length=0
        
    def threaded_open_html_page(self):
        def open_html_file(file_path):
            abs_path = os.path.abspath(file_path)
            url = f'file://{abs_path}'
            webbrowser.open(url)

        if __name__ == '__main__':
            open_html_file(html_path)
            
    def batch_progress(self, val): 
        self.progress_bar.setMaximum(total_image_length)
        self.progress_bar.setValue(val)
        self.batch_processing_function(total_image_length,val)
        
    def batch_processing_function(self,total_image_length,val):
        #self.BatchProcessing_label_path.insert(  f"Processing Batch {val}/{total_image_length}"  )
        text = f"Processing Batch {val}/{total_image_length}"
        self.BatchProcessing_label_path.setText(text)
  
    def  browse_html_file(self):
        filename1, _ = QFileDialog.getOpenFileName(self, 'Select HTML INDEX File', filter='Image Files (*.html)')
        self.html_file_path.setText(filename1)

    def browse_input_folder1(self):
        filename = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.input_folder_path.setText(filename)

    def browse_template_path1(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select Image File', filter='Image Files (*.jpg)')
        self.template_path.setText(filename)
    def browse_output_folder1(self):
        filename = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.output_folder_path.setText(filename)
    def browse_template_path2(self):
        filename1, _ = QFileDialog.getOpenFileName(self, 'Select Image File', filter='Image Files (*.jpg)')
        self.template_path2.setText(filename1)
    def browse_cordinate_path(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select Cordinates file', filter='cordinates file (*.txt)')
        self.cordinate_path.setText(filename)
        
    def browse_db_path1(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select database File', filter='Image Files (*.db)')
        self.db_path.setText(filename)
    def close_window(self):
        global stop
        stop = 1
        self.close()
    def cordinates_point_text_file(self):
        global  barcode_points11_user,DOE_points_user,Paper_code_points_user,Paper_code_2_points_user,semester_points_user_pattern2,\
                    Answer_copy_points_user,center_code_points_user,Paper_No_points_user,college_code_points_user,sitting_points_user,sitting_vertical_points_user,\
                    semester_points_user,additional_points_user,Q1_points_user,Q2_points_user,additional_points_user_pattern2,\
                    Q3_points_user,Q4_points_user,Q5_points_user,Q6_points_user,Q7_points_user,\
                    Q8_points_user,Q9_points_user,Q10_points_user,Q11_points_user,total_points_user,total_digit_user,\
                    Ansidrno_digit_cordinates,Center_Code_digit_cordinates,Paper_No_digit_coordinates,College_Code_digit_cordinates,\
                        Exam_digit_cordinates,Paper_digit_cordinates ,Subject_Code_OCR_cordinates,Subject_Code_user 
        
        cordinates_point_path=self.cordinate_path.text()
        column_field=[]
        pd_ocr_field=[]

        def update_global_variables(data):
            for key, value in data.items():
                if key in globals() and isinstance(globals()[key], list):
                    globals()[key].extend(value)
        def clean_and_format_data( data):
                cleaned_data = {}

                for variable_name, variable_data in data.items():
                    cleaned_list = []

                    for entry in variable_data:
                        # Extract relevant keys and convert values to tuple
                        if all(key in entry for key in ['left', 'top', 'width', 'height']):
                            point_tuple = (
                                entry['left'],
                                entry['top'],
                                entry['width'],
                                entry['height']
                            )
                            cleaned_list.append(point_tuple)

                    cleaned_data[variable_name] = cleaned_list

                return cleaned_data


        with open(cordinates_point_path, "r") as file:
            data_received = json.load(file)

        for key, value in data_received.items():
            if key in globals() and isinstance(globals()[key], list):
                globals()[key].extend(value)

        cleaned_data = clean_and_format_data(data_received)
            # Add this block to create global variables based on cleaned data
        for key, value in cleaned_data.items():
            globals()[key] = value
            if key == 'Barcode':
                barcode_points11_user=[]
                barcode_points11_user.extend(value)
                column_field.append("Barcode")
            if key == 'Ansidrno':
                Answer_copy_points_user=[]
                Answer_copy_points_user.extend(value)
                column_field.append("Ansidrno")

            if key == 'Additional_Copy':
                additional_points_user=[]
                additional_points_user.extend(value)
                column_field.append("Additional_Copy")
            
            if key == 'Center_Code':
                center_code_points_user=[]
                center_code_points_user.extend(value)
                column_field.append("Center_Code")
            
            if key == 'Paper_No':
                Paper_No_points_user=[]
                Paper_No_points_user.extend(value)
                column_field.append("Paper_No")

            if key == 'College_Code':
                college_code_points_user=[]
                college_code_points_user.extend(value)
                column_field.append("College_Code")

            if key == 'Date_of_Exam':
                DOE_points_user=[]
                DOE_points_user.extend(value)
                column_field.append("Date_of_Exam")

            if key == 'Paper_Code':
                Paper_code_points_user=[]
                Paper_code_points_user.extend(value)
                column_field.append("Paper_Code")

            if key == 'Paper_Code_2':
                Paper_code_2_points_user=[]
                Paper_code_2_points_user.extend(value)
                column_field.append("Paper_Code_2")
                    
            if key == 'Sitting_Horizontal':
                sitting_points_user=[]
                sitting_points_user.extend(value)
                column_field.append("Sitting_Horizontal")

            if key == 'Sitting_Vertical':
                sitting_vertical_points_user=[]
                sitting_vertical_points_user.extend(value)
                column_field.append("Sitting_Vertical")

            if key == 'Semester':
                semester_points_user=[]
                semester_points_user.extend(value) 
                column_field.append("Semester")

            if key == 'Additional_Copy_Pattern2':
                additional_points_user_pattern2=[]
                additional_points_user_pattern2.extend(value)
                column_field.append("Additional_Copy_Pattern2")

            if key == 'Semester_Pattern2':
                semester_points_user_pattern2=[]
                semester_points_user_pattern2.extend(value) 
                column_field.append("Semester_Pattern2")

            if key == 'Subject_Code':
                Subject_Code_user=[]
                Subject_Code_user.extend(value) 
                column_field.append("Subject_Code")
                
            if key == 'Q1':
                Q1_points_user=[]
                Q1_points_user.extend(value)
                column_field.append("Q1")
                print("Q1_points_user",Q1_points_user)
            
            if key == 'Q2':
                Q2_points_user=[]
                Q2_points_user.extend(value)
                column_field.append("Q2")
                print("Q2_points_user",Q2_points_user)
                
            if key == 'Q3':
                Q3_points_user=[]
                Q3_points_user.extend(value)
                column_field.append("Q3")
                print("Q3_points_user",Q3_points_user)
              
            if key == 'Q4':
                Q4_points_user=[]
                Q4_points_user.extend(value)
                column_field.append("Q4")
                print("Q4_points_user",Q4_points_user)
                
            if key == 'Q5':
                Q5_points_user=[]
                Q5_points_user.extend(value)
                column_field.append("Q5")
                print("Q5_points_user",Q5_points_user)
            
            if key == 'Q6':
                Q6_points_user=[]
                Q6_points_user.extend(value)
                column_field.append("Q6")
                print("Q6_points_user",Q6_points_user)
              
            if key == 'Q7':
                Q7_points_user=[]
                Q7_points_user.extend(value)
                column_field.append("Q7")
                print("Q7_points_user",Q7_points_user)
            if key == 'Q8':
                Q8_points_user=[]
                Q8_points_user.extend(value)
                column_field.append("Q8")
                print("Q8_points_user",Q8_points_user)
            if key == 'Q9':
                Q9_points_user=[]
                Q9_points_user.extend(value)
                column_field.append("Q9")
                print("Q9_points_user",Q9_points_user)
            if key == 'Q10':
                Q10_points_user=[]
                Q10_points_user.extend(value)
                column_field.append("Q10")
                print("Q10_points_user",Q10_points_user)
            if key == 'Q11':
                Q11_points_user=[]
                Q11_points_user.extend(value)
                column_field.append("Q11")
                print("Q11_points_user",Q11_points_user)
            if key == 'Total_Marks':
                total_points_user=[]
                total_points_user.extend(value) 
                #print("total_points_user",total_points_user)
                column_field.append("Total_Marks")
                print("total_points_user",total_points_user)
                #print(total_points_user)
            if key== "Total_Marks_OCR":
                total_digit_user=[]
                total_digit_user.extend(value) 
                #print("total_digit_user",total_digit_user)
                column_field.append("Total_Marks_OCR")


            if key == 'Ansidrno_OCR':
                Ansidrno_digit_points_user=[]
                Ansidrno_digit_points_user.extend(value)
                Ansidrno_digit_cordinates=Ansidrno_digit_points_user
                pd_ocr_field.append('Ansidrno')

            if key == 'Center_Code_OCR':
                Center_Code_digit_points_user=[]
                Center_Code_digit_points_user.extend(value)
                Center_Code_digit_cordinates=Center_Code_digit_points_user
                pd_ocr_field.append('Center_Code')
              
            if key == 'College_Code_OCR':
                College_Code_digit_points_user=[]
                College_Code_digit_points_user.extend(value)
                College_Code_digit_cordinates=College_Code_digit_points_user
                pd_ocr_field.append('College_Code')
              
            if key == 'Paper_No_OCR':
                Paper_No_digit_points_user=[]
                Paper_No_digit_points_user.extend(value)
                Paper_No_digit_coordinates=Paper_No_digit_points_user
                pd_ocr_field.append('Paper_No')

            
            if key == 'Date_of_Exam_OCR':
                Exam_digit_points_user=[]
                Exam_digit_points_user.extend(value)
                Exam_digit_cordinates=Exam_digit_points_user
                pd_ocr_field.append('Date_of_Exam')
            if key == 'Paper_Code_OCR':
                Paper_digit_points_user=[]
                Paper_digit_points_user.extend(value)
                Paper_digit_cordinates=Paper_digit_points_user
                pd_ocr_field.append('Paper_Code')
            
            if key == 'Paper_Code_2_OCR':
                Paper_digit_points_user=[]
                Paper_digit_points_user.extend(value)
                Paper_digit_cordinates=Paper_digit_points_user
                pd_ocr_field.append('Paper_Code_2')
        
            if key == 'Subject_Code_OCR':
                Subject_Code_OCR_user=[]
                Subject_Code_OCR_user.extend(value)
                Subject_Code_OCR_cordinates=Subject_Code_OCR_user
                pd_ocr_field.append('Subject_Code')

        update_global_variables(cleaned_data)   

        self.columns_platform.setText(', '.join(column_field))
        #print("pd_ocr_field",pd_ocr_field)
        self.columns_platform_ocr.setText(', '.join(pd_ocr_field))  
        del column_field
  
    def process_straight_one_image(self):
        global omr_type_list
        omr_type= self.columns_platform_question.currentText()
        first_string = omr_type
        omr_type_list = [item.strip() for item in first_string.split(',')]
        folder_path= self.input_folder_path.text() 
        template_path=self.template_path.text()
        output_path = os.path.join(folder_path, "output_straight_10_image")
        os.makedirs(output_path, exist_ok=True)
        width=self.image_Width.text()
        height=self.image_Height.text()
        Image_width=int(width)
        Image_height=int(height)
        def Similarity_average_check(straightened_image,template_path ) :
                    template_image=cv2.imread(template_path)
                    
                    if straightened_image is  None:
                        return -1
                    else:
                        SImage_h, SImage_w, _ = straightened_image.shape
                        template_image= cv2.resize(template_image,(SImage_w,SImage_h))
                        similarity1 = cv2.matchTemplate(template_image, straightened_image, cv2.TM_CCOEFF_NORMED)   
                        similarity_percentage1 = similarity1.max() * 100         
                        return similarity_percentage1
        
        def similarity_check2(output_path, template_path, image_count):
            global new_output_path
            similarity_filenames = []
            similarity_scores = []

            for filename in os.listdir(output_path):
                if filename.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(output_path, filename)
                    image = cv2.imread(image_path)
                    similarity_score = Similarity_average_check(image, template_path)
                    similarity_filenames.append(filename)
                    similarity_scores.append(similarity_score)

            if similarity_scores:
                max_similarity_index = similarity_scores.index(max(similarity_scores))
                max_similarity_filename = similarity_filenames[max_similarity_index]

                # Create the new output path if it doesn't exist
                new_output_path = os.path.join(output_path, "out1")
                os.makedirs(new_output_path, exist_ok=True)

                for filename in os.listdir(output_path):
                    if filename.lower().endswith(('.jpg', '.png')) and max_similarity_filename == filename:
                        image_path = os.path.join(output_path, filename)
                        new_image_path = os.path.join(new_output_path, filename)
                        image = cv2.imread(image_path)
                        cv2.imwrite(new_image_path, image)
            else:
                #print("No valid similarity scores.")
                pass

            return new_output_path
        
        
        def similarity_check(folder_path, template_path,yolo_model_pd,omr_type_list,output_path):
                    check = 1
                    total = 0
                    straight_image_list = []
                    total_image_checking=10
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            image_path = os.path.join(folder_path, filename)
                            if check <= 10:
                                text_to_display = f"  Processing ::{check}/{total_image_checking}"
                                self.Processing.setText(text_to_display)
                                self.pbar.setValue(check)
                                straightened_image = straighten_image(image_path,yolo_model_pd,omr_type_list,Image_width,Image_height)
                                straight_image_list.append(straightened_image)
                                sum = Similarity_average_check(straightened_image,template_path)
                                total = total + sum
                                check += 1

                    average = total / 10
                    final_similarity_average = average
                    return final_similarity_average, straight_image_list
        def similarity_check1(straight_image_list, output_path, template_path11, final_similarity_average):
                check = 1
                i = 0
                for straightened_image in straight_image_list:
                    if check <= 10:   
                        similarity_percentage = Similarity_average_check(straightened_image,template_path11)
                        if similarity_percentage > final_similarity_average:
                            output_filename = os.path.join(output_path, f"{i+1}.jpg")
                            cv2.imwrite(output_filename, straightened_image)
                            i += 1
                            check += 1
                image_files = [
                    file for file in os.listdir(output_path) 
                    if os.path.isfile(os.path.join(output_path, file)) and file.lower().endswith('.jpg')
                ]
                image_count=len(image_files)
        
                if image_count> 1 :
                    image_path=''
                    
                    new_image_path=similarity_check2(output_path, template_path11, image_count)
                    for image_file in os.listdir(new_image_path):
                        if image_file.endswith('.jpg'):
                            image_path = os.path.join(new_image_path, image_file)
                
                    QMessageBox.information(self, "Result", f"Straight Image Saved @ : {new_image_path}")
                    self.template_path2 .setText(f'{image_path }')
                    return image_path 
                else:
                    image_path=''
                    filename_straight=[]
                    for image_file in os.listdir(output_path):
                        filename_straight.append(image_file)
                    filename= filename_straight[0]   
                    image_path = os.path.join(output_path, filename)
                    QMessageBox.information(self, "Result", f"Straight Image Saved @ : {output_filename}")
                    self.template_path2 .setText(f'{image_path}')
                    return image_path 
        def find_SI_value(straight_image_list,final_straight_image_path):  

            SI_list=[]
            for straightened_image in straight_image_list:
                similarity_percentage = Similarity_average_check(straightened_image,final_straight_image_path)
                SI_list.append(similarity_percentage)
            total_sum=sum(SI_list)
            average = total_sum / 10
            return average

        
        final_similarity_average, straight_image_list = similarity_check(folder_path, template_path,yolo_model_pd,omr_type_list,output_path)        
        final_straight_image_path=similarity_check1(straight_image_list,output_path, template_path, final_similarity_average)
        print("final_straight_image_path",final_straight_image_path)
        final_similarity_average=find_SI_value(straight_image_list,final_straight_image_path)
        print("final_similarity_average",final_similarity_average)
        if final_similarity_average>35:
            final_similarity_average=int(final_similarity_average)-25
        # elif final_similarity_average>20 and final_similarity_average<=45:
        #     final_similarity_average=int(final_similarity_average)-20 
        else:
            final_similarity_average=int(final_similarity_average)
        QMessageBox.information(self, "Result", f"Average similarity of first 10 images is: {final_similarity_average}")
        self.similarity_per .setText(f'{final_similarity_average}')

    def load_saved_db_config(self):
 
        config_file = "C:\\postgres_config_path\\db_config_pd_marks.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
                self.dbname_lineedit.setText(config.get('dbname', ''))
                self.user_lineedit.setText(config.get('user', ''))
                self.password_lineedit.setText(config.get('password', ''))
                self.host_lineedit.setText(config.get('host', ''))
                self.port_lineedit.setText(config.get('port', ''))
    def btnstate(self,b):
        global check_saved_folder
  
        if b.isChecked() == True:
            check_saved_folder='True'
        else:
            check_saved_folder='False'  

def first_gui():
    
    app = QApplication(sys.argv)
    window = ScriptLauncher()
    window.show()
    app.exec_()

if __name__ == "__main__":
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\BarcodeReaderMutex")
    last_error = ctypes.windll.kernel32.GetLastError()
    if last_error == 183:  # ERROR_ALREADY_EXISTS
        sys.exit(1)
    first_gui()


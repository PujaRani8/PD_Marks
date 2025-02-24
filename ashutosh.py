import ctypes
import sys
import os
import json
import cv2
import math
import imutils
import numpy as np
import time
import re
from PySide2.QtWidgets import QCheckBox,QDialog,  QApplication, QMainWindow, QLabel,QMessageBox ,QVBoxLayout,QHBoxLayout, QWidget, QLineEdit, QProgressBar, QPushButton, QFileDialog, QListWidget, QComboBox
from PySide2.QtGui import QFont 
from PySide2.QtCore import Qt, QThread, Signal
import webbrowser
from ultralytics import YOLO
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sqlite3
import easyocr
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import base64
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from flask_session import Session
import statistics

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Setup logging
logging.basicConfig(filename='image_processing.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Model Configuration
MODEL_CONFIG = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1
}

# Safety Settings of Model
safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings
)

script_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
html_path_anchor=os.path.join(script_dir, "Dependencies_F3","html_anchor_2.html")


################################35######################################
model_path_col35_1 = os.path.join(script_dir, "Dependencies_F3","anchor1_col2.pt")#"35_format_Anchor1_best.pt")
model_path_col35_2 =  os.path.join(script_dir, "Dependencies_F3","anchor_29_f3.pt")#"35_format_Anchor2_best.pt")

############################ 44 ###################################
model_path_col44_1 = os.path.join(script_dir, "Dependencies_F3","anchor1_col2.pt")#"44_format_Anchor_1_best.pt")
model_path_col44_2 =  os.path.join(script_dir, "Dependencies_F3","44_format_Anchor_2_best_F2.pt")#"44_format_Anchor_2_best.pt")

#################################### 57 ####################################
model_path_col57_1 = os.path.join(script_dir, "Dependencies_F3","anchor1_col2.pt")#"57_format_Anchor_1_best.pt")
model_path_col57_2 =  os.path.join(script_dir, "Dependencies_F3","57_format_Anchor_2_best.pt")

############################# 45 #######################################################
model_path_col45_1 = os.path.join(script_dir, "Dependencies_F3","anchor1_col2.pt")#"tr45_anchor1_F1.pt")#"45_format_Anchor_1_best.pt")
model_path_col45_2 =  os.path.join(script_dir, "Dependencies_F3","anchor2_tr45.pt")#"45_format_Anchor_2_best.pt")
################################ 46 ####################
model_path_col46_1 = os.path.join(script_dir, "Dependencies_F3","anchor1_col2.pt")#"tr45_anchor1_F1.pt")#"45_format_Anchor_1_best.pt")
model_path_col46_2 =  os.path.join(script_dir, "Dependencies_F3","tr46_anchor2_F1.pt")#"45_format_Anchor_2_best.pt")
################################# 50 ##########################
model_path_col50_1 = os.path.join(script_dir, "Dependencies_F3","anchor1_col2.pt")#"col50_anchor1_F1.pt")#"50_format_Anchor_1_best.pt")
model_path_col50_2 =  os.path.join(script_dir, "Dependencies_F3","col50_anchor2_F1.pt")#"50_format_Anchor_2_best.pt")
################################### 52########################################################
model_path_col52_1 = os.path.join(script_dir, "Dependencies_F3","anchor1_col2.pt")#"52_format_Anchor1_best.pt")
model_path_col52_2 =  os.path.join(script_dir, "Dependencies_F3","52_format_Anchor2_best.pt")
################################################################
model_col_name= os.path.join(script_dir, "Dependencies_F3","collge_name_yolo.pt")
model_column4= os.path.join(script_dir, "Dependencies_F3","column4_F4.pt")#"column4_F3.pt")#column4_F2.pt")#column4_F1.pt")


yolo_model_col35_1 = YOLO(model_path_col35_1)
yolo_model_col35_2 = YOLO(model_path_col35_2)

yolo_model_col44_1 = YOLO(model_path_col44_1)
yolo_model_col44_2 = YOLO(model_path_col44_2)

yolo_model_col57_1 = YOLO(model_path_col57_1)
yolo_model_col57_2 = YOLO(model_path_col57_2)

yolo_model_col45_1 = YOLO(model_path_col45_1)
yolo_model_col45_2 = YOLO(model_path_col45_2)

yolo_model_col46_1 = YOLO(model_path_col46_1)
yolo_model_col46_2 = YOLO(model_path_col46_2)


yolo_model_col50_1 = YOLO(model_path_col50_1)
yolo_model_col50_2 = YOLO(model_path_col50_2)

yolo_model_col52_1 = YOLO(model_path_col52_1)
yolo_model_col52_2 = YOLO(model_path_col52_2)
yolo_college_name = YOLO(model_col_name)


yolo_column4 = YOLO(model_column4)
model_anchor1={}


########################## prompt according to pattern#####################

########################## prompt according to pattern#####################
prompt_printed="""
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 1 to Column 13.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

                Column 1 is Roll No.
                
                Column 2 is Reg No.
                
                Column 3 is Name of the examiniees.
                For Column 4 is General Subjects.Take only if alphabetic Data.

                If the whole slice is blank marks it '-'
                
            IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
"""

prompt_52_1 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 4 to Column 9.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

	        For Column 4 is General Subjects.Take only if alphabetic Data.

            For Column 5 , marks out of 100.(if column is blank, use '-').

            For Column 6 , marks out of 100.(if column is blank, use '-').

            For Column 7 , marks out of 200. (if column is blank, use '-').

            For Column 8 , is Subject Name. Take only if alphabetic Data.

            For Column 9 , marks out of 50.(if column is blank, use '-').

            For Column 10 (If present this column) , marks out of 50.(if column is blank, use '-').

            For Column 11 (If present this column) , marks out of 100. (if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            
            """



prompt_52_2 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 10 to Column 16.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 8 (If present this column) , is Subject Name. Take only if alphabetic Data.

            For Column 9 (If present this column) , marks out of 50.(if column is blank, use '-').

            For Column 10 , marks out of 50.(if column is blank, use '-').

            For Column 11  , marks out of 100. (if column is blank, use '-').

            For Column 12 , marks out of 50.(if column is blank, use '-').

            For Column 13 , marks out of 50.(if column is blank, use '-').

            For Column 14 , marks out of 100. Mostly column is blank so carefully check. if column is blank, use '-'
            
            Column 15 is Subject Name. Take only if alphabetic Data.

            For Column 16 , theory marks out of 100. (if column is blank, use '-').

            For Column 17 (If present this column), practical marks out of 25. (if column is blank, use '-').

            For Column 18 (If present this column), theory marks out of 50. (if column is blank, use '-').

            Format the output as follows, without any extra details:

            verfy the sum -  if not match then write 'RC'

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            """


prompt_52_3 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 17 to Column 23
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. Ensure the data aligns exactly with             	    Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            Column 15 (If present this column) is Subject Name. Take only if alphabetic Data.

            For Column 16 (If present this column)  , theory marks out of 100. (if column is blank, use '-').

            For Column 17 , practical marks out of 25. (if column is blank, use '-').

            For Column 18 , theory marks out of 50. (if column is blank, use '-').

            For Column 19 , practical marks out of 25. (if column is blank, use '-').

            For Column 20 , Total  Marks out of 150.(if column is blank, use '-').

            For Column 21 , Total  marks  out of 50. (if column is blank, use '-').

            For Column 22 , marks out of 200. (if column is blank, use '-').

            Column 23 is Subject Name. Take only if alphabetic Data.

            For Column 24 (If present this column) , marks out of 100.(if column is blank, use '-').

            For Column 25 (If present this column) , marks out of 100 column is blank, use '-').



            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """


prompt_52_4 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 24 to Column 36
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 22 (If present this column) , marks out of 200. (if column is blank, use '-').

            Column 23 (If present this column) is Subject Name. Take only if alphabetic Data.

            For Column 24 , marks out of 100.(if column is blank, use '-').

            For Column 25 , marks out of 100 column is blank, use '-').

            For Column 26 , Total marks out of 200. (if column is blank, use '-').

            For Column 27 , theory marks out of 75. Two Digit Number(if column is blank, use '-').

            For Column 28 , theory marks out of 75.(if column is blank, use '-').

            For Column 29 , Practical marks out of 50. (if column is blank, use '-').

            For Column 30 , theory marks out of 200.(if column is blank, use '-').

            For Column 31 , theory marks out of 75.(if column is blank, use '-').

            For Column 32 , theory marks out of 75.(if column is blank, use '-').

            For Column 33 , Practical marks out of 50. (if column is blank, use '-').

            For Column 34 , Total marks out of 200.(if column is blank, use '-').

            For Column 35 , marks out of 100.(if column is blank, use '-').

            For Column 36 , marks out of 100 column is blank, use '-').

            For Column 37 (If present this column) , marks out of 100. (if column is blank, use '-').

            For Column 38 (If present this column) , marks out of 100.(if column is blank, use '-').

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            """

prompt_52_5 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 37 to Column 52
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 


            For Column 35 (If present this column) , marks out of 100.(if column is blank, use '-').

            For Column 36  (If present this column) , marks out of 100 column is blank, use '-').

            For Column 37  , marks out of 100. (if column is blank, use '-').

            For Column 38  , marks out of 100.(if column is blank, use '-').

            For Column 39 , marks out of 100. Mostly column is blank so carefully check. if column is blank, use '-'

            For Column 40 , Total marks out of 400.(if column is blank, use '-').

            For Column 41 , This column is mostly blank so use '-' for it. and if got value marks out of 100.(if column is blank, use '-').

            For Column 42 , marks out of 600.(if column is blank, use '-').

            For Column 43 , marks out of 200. Mostly column is blank so carefully check. if column is blank, use '-'

            For Column 44 , Total marks out of 800.(if column is blank, use '-').

            Not to mix column 45 and 46 strictly.

            For Column 45 , marks out of 100. Numeric 2 Digit Number. (if column is blank, use '-').

            For Column 46 , Grand Total marks out of 1500. Numeric 3 Digit Number. (if column is blank, use '-').
            
            Column 47 is Roll No.

            Column 48 is For 1st Division in Roman.
            Column 49 is For 2nd Division in Roman.
            Column 50 , is for Distinction.

            Column 51, is for Fail.

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            """

prompt_35_1 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 4 to Column 9
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
        
            For Column 4 , marks out of 100.(if column is blank, use '-').
            For Column 5 , marks out of 100.(if column is blank, use '-').
            For Column 6 , marks out of 200. (if column is blank, use '-').

            For Column 7 , marks out of 50.(if column is blank, use '-').
            For Column 8 , marks out of 50.(if column is blank, use '-').
            For Column 9 , marks out of 100. (if column is blank, use '-').

            For Column 10(If present this column), is subject name.(Alphabetic Data)
            For Column 11 (If present this column), marks out of 50. (if column is blank, use '-').

            Format the output as follows, without any extra details:

            verfy the sum -  if not match then write 'RC'
            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """



prompt_35_2 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 10 to Column 16
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 8(If present this column) , marks out of 50.(if column is blank, use '-').
            For Column 9(If present this column) , marks out of 100. (if column is blank, use '-').

            For Column 10, is subject name.(Alphabetic Data)

            For Column 11, marks out of 50. (if column is blank, use '-').
            For Column 12 , marks out of 50.(if column is blank, use '-').

            For Column 13 , marks out of 100. (if column is blank, use '-').
    
            For Column 14 , marks out of 100. (if column is blank, use '-').
            
            For Column 15, marks out of 100. (if column is blank, use '-').

            For Column 16, marks out of 100. (if column is blank, use '-').

            For Column 17(If present this column), marks out of 300. (if column is blank, use '-').
            
            For Column 18(If present this column), marks out of 100. (if column is blank, use '-').

            Format the output as follows, without any extra details:

            verfy the sum -  if not match then write 'RC'

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """


prompt_35_3 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 31 to Column 23
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 15(If present this column), marks out of 100. (if column is blank, use '-').

            For Column 16(If present this column), marks out of 100. (if column is blank, use '-').

            For Column 17, marks out of 300. (if column is blank, use '-').
            
            For Column 18, marks out of 100. (if column is blank, use '-').

            For Column 19, marks out of 100. (if column is blank, use '-').
            For Column 20, marks out of 100. (if column is blank, use '-').

            For Column 21, marks out of 300. (if column is blank, use '-').


            For Column 22, marks out of 100. (if column is blank, use '-').


            For Column 23 , Group Name(Alphabetic Data). Take only if alphabetic Data.

            For Column 24(If present this column) , marks data  out of 100.(if column is blank, use '-').
            For Column 25(If present this column) , marks data out of 50. (if column is blank, use '-').

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            """


prompt_35_4 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 24 to Column 30
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 22(If present this column), marks out of 100. (if column is blank, use '-').
            For Column 23(If present this column), Group Name(Alphabetic Data). Take only if alphabetic Data.

            For Column 24 , marks data  out of 100.(if column is blank, use '-').
            For Column 25 , marks data out of 50. (if column is blank, use '-').

            For Column 26 , Total marks out of 100.(if column is blank, use '-').
            For Column 27 , marks out of 200.(if column is blank, use '-').

            For Column 28 , marks out of 1200.(if column is blank, use '-').

            For Column 29 , Roll No.
            For Column 30 , 1st Division in Roman.

            For Column 31(If present this column), 2nd Division in Roman.
            For Column 32 (If present this column), 3rd Division in Roman.


            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """



prompt_35_5 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 31 to Column 35
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.
            

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 


            For Column 29(If present this column) , Roll No.
            For Column 30 (If present this column), 1st Division in Roman.

            For Column 31 , 2nd Division in Roman.
            For Column 32 , 3rd Division in Roman.
            
            Column 33 Distinction.

            Column 34 is For Fail Data
            Column 35 is For Remark.

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """
            


prompt_44_1 = """

            The first row shows system-generated numbers representing column names, ranging from Column 4 to Column 9 (Sometimes 10).
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

	        Column 4 is General Subjects.Take only if alphabetic Data.

            For Column 5 , is for Composition Data (alphabetic Data).(if column is blank, use '-').

            For Column 6 , Subsidiary subjects(alphabetic Data).(if column is blank, use '-').

            For Column 7 , marks out of 100. (if column is blank, use '-').

            For Column 8 , marks out of 100. (if column is blank, use '-').

            For Column 9 , marks out of 200.(if column is blank, use '-').

            For Column 10(If prsent) , marks out of 50.(if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            
            """


prompt_44_2 = """

            The first row shows system-generated numbers representing column names, ranging from Column 10 to Column 16.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 9(If prsent) , marks out of 200.(if column is blank, use '-'). 

            For Column 10 , marks out of 50.(if column is blank, use '-').

            For Column 11 , marks out of 50. (if column is blank, use '-').

            For Column 12 , marks out of 100.(if column is blank, use '-').

            For Column 13 , is Subject Name.

            For Column 14 , marks out of 50. (if column is blank, use '-').

            For Column 15 , marks out of 500. (if column is blank, use '-').

            For Column 16 , marks out of 100.(if column is blank, use '-').

            For Column 17(If present) , marks out of 100.(if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            
            """

prompt_44_3 = """

            The first row shows system-generated numbers representing column names, ranging from Column 14 to Column 25.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 16 (If present) , marks out of 100.(if column is blank, use '-').

            For Column 17 , marks out of 100.(if column is blank, use '-').

            For Column 18 , marks out of 100. (if column is blank, use '-').

            For Column 19 , marks out of 100.(if column is blank, use '-').

            For Column 20 , marks out of 100.(if column is blank, use '-').

            For Column 21 , marks out of 100.(if column is blank, use '-').

            For Column 22 ,  this column is mostly blank, use '-'
    
            For Column 23 , this column is mostly blank, use '-'

            For Column 24 (If present) , marks out of 100.(if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """


prompt_44_4 = """

            The first row shows system-generated numbers representing column names, ranging from Column 24 to Column 35.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 23 (If present), this column is mostly blank, use '-'
            
            For Column 24 , marks out of 100.(if column is blank, use '-').
            
            For Column 25 , marks out of 100.(if column is blank, use '-').
            
            For Column 26 , marks out of 200. (if column is blank, use '-').
            
            For Column 27 , marks out of 100. (if column is blank, use '-').

            For Column 28 , marks out of 100.(if column is blank, use '-').

            For Column 29 , marks out of 200.(if column is blank, use '-').

            For Column 30 , marks out of 100. (if column is blank, use '-').

            For Column 31 , marks out of 100.(if column is blank, use '-').

            For Column 32 , marks out of 100.(if column is blank, use '-').

            For Column 33 , marks out of 100.(if column is blank, use '-').

            For Column 34 ,  marks out of 400.(if column is blank, use '-').
    
            For Column 35 , marks out of 800.(if column is blank, use '-').

            For Column 36 (if present): Marks out of 100 (use `-` if blank).

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            """


prompt_44_5 = """
        The table consists of two rows:
        - First Row**: System-generated column names (Column 36 to Column 44).
        - Second Row**: Handwritten data (marks, subject names, or student names).

        ### Instructions:
        1. Data Verification**:
        - Check ach column carefully, ensuring that the handwritten data in the second row corresponds to the correct column number in the first row.
        - Use vertical lines for alignment and avoid shifting data between columns.

        2. Handling Missing or Incorrect Data**:
        - If data is missing or incorrect, mark it with `-`. Do not move data from another column to fill the gap.
        - For columns requiring calculations (e.g., sums of other columns), if any input column is blank, the result column must be marked as `-`.

        3. **Output Format:
        - Use the format: `"Column_No" - Data`. 
            Example: `Column_1 - 23`.
        - Do not include any additional text or symbols (e.g., no `+` in calculations).

        4. Column Details:

        - For Column 35 (if present) : marks out of 800.(if column is blank, use '-').  

        - For Column 36: Marks out of 100 (use `-` if blank).

        - Column 37: Grand Total marks out of 1600 (numeric, 3-digit; use `-` if blank).

        - Column 38: Roll Number.

        - column 39: Roman numeral for 1st class.

        - Column 40: Roman numeral for 2nd class.

        - Column 41: Distinction.

        - Column 42: Fail.

        - Column 43: Marksheet Number.

        - Column 44: Teacher’s remark (2-3 words).

        5. Special Cases**:
        - If the entire row (slice) is blank, mark all columns as `-`.

        ### Key Points:
        - Do not shift or adjust data between columns.
        - Follow the exact output format: `"Column_No" - Data`. 
        Example: `Column_1 - 23`.

        """


prompt_46_1 = """

            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 4 to Column 11.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
    
    CHECK CAREFULLY ALL COLUMN HEADLINES COLUMN NUMBER AND DATA SHOULD BE MATCHED.

            For Column 4, is Subject Name. Take only if alphabetic Data.

            For Column 5 , marks out of 100.(if column is blank, use '-').

            For Column 6  , marks out of 100. (if column is blank, use '-').

        COLUMN 7 IS SUM OF COLUMN 5 AND COLUMN 6 SO TAKE CAREFULLY. NOT ALWAYAS SO KEEPCOLUMN 7 VALUE AS IT IS EVEN IF COLUMN 5 AND 6 IS BLANK OR SUM IS NOT MATCHING.
            For Column 7 , marks out of 200.(if column is blank, use '-').

            For Column 8  , marks out of 50. (if column is blank, use '-').

            For Column 9 , marks out of 50.(if column is blank, use '-').

            For Column 10, marks out of 100.(if column is blank, use '-').

            For Column 11, is Subject Name. Take only if alphabetic Data. (STRICTLY CHECK THIS COLUMN WELL)

            For Column 12, marks out of 50.(if column is blank, use '-').

            For Column 13(IF PRESENT), marks out of 50. (if column is blank, use '-').

            For Column 14(IF PRESENT), marks out of 100. (if column is blank, use '-').

            Format the output as follows, without any extra details:

            verfy the sum -  if not match then write 'RC'

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            
            """


prompt_46_2 = """

            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 12 to Column 19.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 


    CHECK CAREFULLY ALL COLUMN HEADLINES COLUMN NUMBER AND DATA SHOULD BE MATCHED.

            For Column 10(IF PRESENT), marks out of 100.(if column is blank, use '-').

            For Column 11(IF PRESENT), is Subject Name. Take only if alphabetic Data.

            For Column 12, marks out of 50.(if column is blank, use '-').

            For Column 13, marks out of 50. (if column is blank, use '-').

            Check strictly in column 14 only numeric data
            For Column 14 , marks out of 100. (if column is blank, use '-').

            For Column 15 , is Subject Name. Take only if alphabetic Data.

            Check carefully column check Column number and data alighned.

            For Column 16 , marks out of 100.(if column is blank, use '-').

            For Column 17 , marks out of 100.(if column is blank, use '-').
        
            For Column 18, marks out of 100.(if column is blank, use '-').

            For Column 19, marks out of 300.(if column is blank, use '-').

            For Column 20(IF PRESENT), blank column , use '-' for blank.

            For Column 21(IF PRESENT), is Subject Name. Take only if alphabetic Data.

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            
            """

prompt_46_3 = """

            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 20 to Column 29.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 18(IF PRESENT), marks out of 100.(if column is blank, use '-').

            For Column 19(IF PRESENT), marks out of 300.(if column is blank, use '-').

            For Column 20 , blank column , use '-' for blank.

            For Column 21 , is Subject Name. Take only if alphabetic Data.

            For Column 22 , marks out of 100.(if column is blank, use '-').

        STRICLTLY CHECK COLUMN 23 AND 25 MOSTLY BLANK SO CHECK IT CAREFULLY.   

            For Column 23 , marks out of 50.(if column is blank, use '-').

            For Column 24, marks out of 100.(if column is blank, use '-').

            For Column 25, marks out of 50.(if column is blank, use '-').

            For Column 26 , marks out of 100.(if column is blank, use '-').

            For Column 27 , marks out of 50.(if column is blank, use '-').

            For Column 28 , marks out of 300.(if column is blank, use '-').

            For Column 29 , marks out of 75.(if column is blank, use '-').

            For Column 30(IF PRESENT), is Subject Name. Take only if alphabetic Data.

            For Column 31(IF PRESENT), marks out of 100.(if column is blank, use '-').

            If the whole slice is blank marks it '-'

            """


prompt_46_4 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 30 to Column 38.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 28(IF PRESENT), marks out of 300.(if column is blank, use '-').

            For Column 29(IF PRESENT), marks out of 75.(if column is blank, use '-').

            For Column 30 , is Subject Name. Take only if alphabetic Data.

            For Column 31, marks out of 100.(if column is blank, use '-').
            
            For Column 32, marks out of 100.(if column is blank, use '-').
    
        CHECK WELL COLUMN 33 IS MOSTLY BLANK COLUMN.
            For Column 33, marks out of 50.(if column is blank, use '-').

            For Column 34, marks out of 100.(if column is blank, use '-').
    
            For Column 35, marks out of 50.(if column is blank, use '-').

        Stricly check column 36 AND 37, mostly 36 having 3 digit and 38 having 2 digit VALUES dont mix it.)

            For Column 36, Total Marks OUT OF 200.(if column is blank, use '-').
    
            For Column 37, marks out of 100.(if column is blank, use '-').

            cOLUMN 38 It is a last value of this row so check it carefully. 3 digit numeric value.

            For Column 38, marks out of 1000.(if column is blank, use '-').

            For Column 39(IF PRESENT), Roll No. (LENGHTS IS 7 DIGIT)
    
            For Column 40(IF PRESENT), 1st Division in Roman..

            If the whole slice is blank marks it '-'

            """


prompt_46_5 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 39 to Column 46.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
    
            For Column 37(IF PRESENT), marks out of 100.(if column is blank, use '-').
            For Column 38(IF PRESENT), marks out of 1000.(if column is blank, use '-').
   
            For Column 39, Roll No. (LENGHTS IS 7 DIGIT)
    
            For Column 40, 1st Division in Roman..

            For Column 41, 2nd Division in Roman.

            For Column 42, 3rd Division in Roman.

            For Column 43, Is for Distinction.

            For Column 44, is for Fail.

            For Column 45, Marksheet No.

            For Column 46, Remark.
    
            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
        

        """



prompt_50_1 = """
            The first row shows system-generated numbers representing column names, ranging from Column 4 to Column 9.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            
	        Column 4 is General Subjects.
            
            For Column 5 , marks out of 100.(if column is blank, use '-').

            For Column 6 , marks out of 100.(if column is blank, use '-').

            For Column 7 , marks out of 200. (if column is blank, use '-').

            For Column 8 , marks out of 50.(if column is blank, use '-').

            For Column 9 , marks out of 50.(if column is blank, use '-').
            
            For Column 10 , marks out of 100. (if column is blank, use '-').

            For Column 11(IF PRESENT) ,subject Name.

            For Column 12(IF PRESENT) , marks out of 50.(if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """

prompt_50_2 = """
             The first row shows system-generated numbers representing column names, ranging from Column 10 to Column 16.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            
            For Column 8(IF PRESENT), marks out of 50.(if column is blank, use '-').

            For Column 9(IF PRESENT), marks out of 50.(if column is blank, use '-').
            
            For Column 10 , marks out of 100. (if column is blank, use '-').

            For Column 11 ,subject Name.
            
            For Column 12, marks out of 50.(if column is blank, use '-').

            For Column 13 , marks out of 50.(if column is blank, use '-').

            For Column 14 , marks out of 100. (if column is blank, use '-').
            
            Column 15 is Subject Name.

            For Column 16 , theory marks out of 100. (if column is blank, use '-').

            For Column 17(IF PRESENT) , practical marks out of 25. (if column is blank, use '-').

            For Column 18(IF PRESENT) , theory marks out of 50. (if column is blank, use '-').


            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """


prompt_50_3 = """
            The first row shows system-generated numbers representing column names, ranging from Column 17 to Column 23.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            Column 15(IF PRESENT),  is Subject Name.

            For Column 16(IF PRESENT) , theory marks out of 100. (if column is blank, use '-').

            For Column 17, practical marks out of 25. (if column is blank, use '-').

            For Column 18, theory marks out of 100. (if column is blank, use '-').

            For Column 19 , practical marks out of 25. (if column is blank, use '-').

            For Column 20 , Total theory Marks out of 100.(if column is blank, use '-').

            For Column 21 , Total practical marks  out of 25. (if column is blank, use '-').

            For Column 22 , theory + practical marks out of 225. Sum of column 20+21 (if column is blank, use '-').

            Column 23 is , practical marks  out of 75. (if column is blank, use '-').

            For Column 24(IF PRESENT) , is Subject Name.

            For Column 25(IF PRESENT) , practical marks  out of 75. (if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """


prompt_50_4 = """
            The first row shows system-generated numbers representing column names, ranging from Column 24 to Column 36.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            For Column 22(IF PRESENT) , theory + practical marks out of 225. Sum of column 20+21 (if column is blank, use '-').

            Column 23 is(IF PRESENT) , practical marks  out of 75. (if column is blank, use '-').

            For Column 24 , is Subject Name.

            For Column 25, practical marks  out of 75. (if column is blank, use '-').

            For Column 26 ,practical marks out of 25. (if column is blank, use '-').

            For Column 27 , theory marks out of 75.(if column is blank, use '-').

            For Column 28 , theory marks out of 25.(if column is blank, use '-').

            For Column 29 , Practical marks out of 75. (if column is blank, use '-').

            For Column 30 ,  marks out of 25.(if column is blank, use '-').

            For Column 31 , Total marks out of 225.(if column is blank, use '-').

            For Column 32 , theory marks out of 75.(if column is blank, use '-').

            For Column 33 , is Subject Name.

            For Column 34 , theory marks out of 75.(if column is blank, use '-').

            For Column 35 , practical marks out of 25.(if column is blank, use '-').

            For Column 36 , theory marks out of 75.(if column is blank, use '-').

            For Column 37(IF PRESENT) , marks out of 25. (if column is blank, use '-').

            For Column 38(IF PRESENT) , marks out of 75.(if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
            """

prompt_50_5 = """
            The first row shows system-generated numbers representing column names, ranging from Column 37 to Column 50.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

    
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 
        Total 14 column are there carefully check all column Dont Misplaced Columns.

            For Column 35(IF PRESENT) , practical marks out of 25.(if column is blank, use '-').

            For Column 36(IF PRESENT) , theory marks out of 75.(if column is blank, use '-').

        Carefully check the data dont misplaced the column focus on out of marks and headline column number.

            For Column 37 , marks out of 25. (if column is blank, use '-').

            For Column 38, marks out of 75.(if column is blank, use '-').

            For Column 39 , marks out of 25.(if column is blank, use '-').

            For Column 40 , Total marks out of 225.(if column is blank, use '-'). (Mostly 3 digit numeric data)

            For Column 41 , marks out of 75.(if column is blank, use '-').

            For Column 42 , marks out of 100.(if column is blank, use '-').

            For Column 43 , marks out of 1200. (if column is blank, use '-'). (Mostly 3 digit numeric data)

            For Column 44 , is Roll No.

            Column 45 is For 1st Division in Roman.
            Column 46 is For 2nd Division in Roman.
            Column 47 is For 2nd Division in Roman.

            Column 48 , is for Distinction.

            Column 19 , is fail data

            Column 50 , is for Remark.

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            """

prompt_57_1 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 1 to Column 15.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

	        Column 4 is General Subjects.
            
            For Column 5 , marks out of 100. Mostly column is blank so carefully check. if column is blank, use '-'

            For Column 6 , Subject Name. Mostly column is blank so carefully check. if column is blank, use '-'

            For Column 7 , marks out of 100. (if column is blank, use '-').

            For Column 8 , marks out of 100. (if column is blank, use '-').

            For Column 9 , marks out of 200. (if column is blank, use '-').

            For Column 10 , is for Subject Name.(if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else. 

            """



prompt_57_2 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 16 to Column 29.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_16 - 23,  dont write anything else. 
            
            For Column 11 , marks out of 50. (if column is blank, use '-').

            For Column 12 , marks out of 50.(if column is blank, use '-').

            For Column 13 , marks out of 100.(if column is blank, use '-').

            For Column 14 , marks out of 50.(if column is blank, use '-').

            For Column 15 , marks out of 50.(if column is blank, use '-').
            For Column 16 , marks out of 100. (if column is blank, use '-').

            For Column 17 , is for Subject Name. Check Column Carefully. (if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_16 - 23,  dont write anything else. 
            """


prompt_57_3 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 30 to Column 43.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_30 - 23,  dont write anything else. 
            
        For Column 18 , marks out of 75. (if column is blank, use '-').

            For Column 19 , marks out of 25. (if column is blank, use '-').

            For Column 20 , marks out of 75. (if column is blank, use '-').

            For Column 21 , marks out of 25. (if column is blank, use '-').

            For Column 22 , marks out of 150.(if column is blank, use '-').

            For Column 23 , marks out of 50.(if column is blank, use '-').

            For Column 24 , is for Subject Name.(if column is blank, use '-').

            For Column 25 , marks out of 75.(if column is blank, use '-').

            For Column 26 , marks out of 25.(if column is blank, use '-').
            
            For Column 27 , marks out of 75.(if column is blank, use '-').

            For Column 28 , marks out of 25.(if column is blank, use '-').

            For Column 29 , marks out of 150.(if column is blank, use '-').

            For Column 30 , marks out of 25.(if column is blank, use '-').

            If the whole slice is blank marks it '-'
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_30 - 23,  dont write anything else. 
            """


prompt_57_4 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 44 to Column 57.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_44 - 23,  dont write anything else. 
            
            For Column 31 , marks out of 75. (if column is blank, use '-').

            For Column 32 , marks out of 75. (if column is blank, use '-').

            For Column 33 , marks out of 50. (if column is blank, use '-').

            For Column 34 , marks out of 150. (if column is blank, use '-').

            For Column 35 , marks out of 50.(if column is blank, use '-').
            
            For Column 36 , marks out of 50. (if column is blank, use '-').

            For Column 37 , marks out of 150. (if column is blank, use '-').

            For Column 36 , marks out of 50. (if column is blank, use '-').

            For Column 37 , marks out of 150. (if column is blank, use '-').

            For Column 38 , marks out of 50.(if column is blank, use '-').

            For Column 39 , marks out of 150.(if column is blank, use '-').
            
            For Column 40 , marks out of 50.(if column is blank, use '-').

            
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_44 - 23,  dont write anything else. 
            """


prompt_57_5 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 44 to Column 57.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_44 - 23,  dont write anything else. 

            For Column 41 , marks out of 75.(if column is blank, use '-').

            For Column 42 , marks out of 75.(if column is blank, use '-').

            For Column 43 , marks out of 75.(if column is blank, use '-').
            For Column 44 , marks out of 75.(if column is blank, use '-').

            For Column 45 , Total marks out of 400. (if column is blank, use '-').

            For Column 46 , marks out of 100. (if column is blank, use '-').

            For Column 47 , marks out of 600. (if column is blank, use '-').

            For Column 48 , marks out of 200. Mostly column is blank so carefully check. if column is blank, use '-'

            For Column 49 , marks out of 800.(if column is blank, use '-'). Check this column carefully.
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_44 - 23,  dont write anything else. 
            """

prompt_57_6 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 44 to Column 57.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. 
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number 
            with the data.

            Instructions:
            
            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_44 - 23,  dont write anything else. 
            
            For Column 50 , marks out of 100. (if column is blank, use '-').

            For Column 51 , Grand Totak marks out of "1500". (if column is blank, use '-'). Check this column carefully. Important Column.

            Column 52 is Roll No.

            Column 53 is For 1st Division in Roman.
            Column 54 is For 2nd Division in Roman.

            Column 55 , is for Distinction.

            Column 56, is for Fail.

            For Column 57, Remark is written by teacher in 2-3 words.
            
	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_44 - 23,  dont write anything else. 
            """



prompt_45_1 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 4 to Column 11.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column.
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number
            with the data.

            Instructions:

            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.

            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.
        Carefully check the data dont misplaced the column focus on out of marks and headline column number.

	        

            For Column 5 , marks out of 100.(if column is blank, use '-').

            For Column 6 , marks out of 100.(if column is blank, use '-').

            For Column 7 , marks out of 200. (if column is blank, use '-').

            For Column 8 , is Subject Name. Take only if alphabetic Data.

            For Column 9 , marks out of 50.(if column is blank, use '-').

            For Column 10 , marks out of 50.(if column is blank, use '-').

            For Column 11 , marks out of 100. (if column is blank, use '-').

            For Column 12 (If present this column) , marks out of 50.(if column is blank, use '-').

            For Column 13 (If present this column) , marks out of 50.(if column is blank, use '-').

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.

            """



prompt_45_2 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 12 to Column 20.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column.
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number
            with the data.

            Instructions:

            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.
        Carefully check the data dont misplaced the column focus on out of marks and headline column number.


            For Column 10 (If present this column), marks out of 50.(if column is blank, use '-').

            For Column 11 (If present this column), marks out of 100. (if column is blank, use '-').

            For Column 12 , marks out of 50.(if column is blank, use '-').

            For Column 13 , marks out of 50.(if column is blank, use '-').

            For Column 14 , marks out of 100. Mostly column is blank so carefully check. if column is blank, use '-'

            Column 15, is Subject Name. Take only if alphabetic Data.

            For Column 16 , theory marks out of 100. (if column is blank, use '-').

            For Column 17  (If present this column) , practical marks out of 25. (if column is blank, use '-').

            For Column 18  (If present this column), theory marks out of 50. (if column is blank, use '-').

            For Column 19 , practical marks out of 25. (if column is blank, use '-').

            For Column 20 , Total  Marks out of 150.(if column is blank, use '-').

            For Column 21 (If present this column), Total  marks  out of 50. (if column is blank, use '-').

            For Column 22 (If present this column), marks out of 200. (if column is blank, use '-').

            Column 23 , is Subject Name. Take only if alphabetic Data.

            For Column 24 (If present this column) , marks out of 100.(if column is blank, use '-').

            Format the output as follows, without any extra details:

            verfy the sum -  if not match then write 'RC'

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.

            """


prompt_45_3 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 21 to Column 29.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column. Ensure the data aligns exactly with             	    Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column.
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number
            with the data.

            Instructions:

            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.
        Carefully check the data dont misplaced the column focus on out of marks and headline column number.

            For Column 19 (If present this column), practical marks out of 25. (if column is blank, use '-').

            For Column 20 (If present this column), Total  Marks out of 150.(if column is blank, use '-').

            For Column 21 , Total  marks  out of 50. (if column is blank, use '-').

            For Column 22 , marks out of 200. (if column is blank, use '-').

            Column 23 , is Subject Name. Take only if alphabetic Data.

            For Column 24  , marks out of 100.(if column is blank, use '-').

            For Column 25 , marks out of 50 column is blank, use '-').

        * Do check carefully column 26 & 27.

            For Column 26 , Total marks out of 50. (if column is blank, use '-').

            For Column 27 , marks out of 50. Two Digit Number(if column is blank, use '-').

            For Column 28 , theory marks out of 200.(if column is blank, use '-').

            For Column 29 , Practical marks out of 100. (if column is blank, use '-').

            For Column 30 (If present this column), is Subject Name. Take only if alphabetic Data.

            For Column 31 (If present this column), theory marks out of 100.(if column is blank, use '-').

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.
            """


prompt_45_4 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 30 to Column 38.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column.
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number
            with the data.

            Instructions:

            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.
        Carefully check the data dont misplaced the column focus on out of marks and headline column number.


            For Column 28 (If present this column), theory marks out of 200.(if column is blank, use '-').

            For Column 29 (If present this column), Practical marks out of 100. (if column is blank, use '-').

            For Column 30 , is Subject Name. Take only if alphabetic Data.

            For Column 31 , theory marks out of 100.(if column is blank, use '-').

            For Column 32 , theory marks out of 50.(if column is blank, use '-').

            For Column 33 , theory marks out of 50.(if column is blank, use '-').

            For Column 34 , Practical marks out of 50.(if column is blank, use '-').

            For Column 35 , marks in 3 Digit(Mumeric Value).(if column is blank, use '-').

            For Column 36 , marks out of 100 column is blank, use '-').

            For Column 37 , Grand Total marks out of 1200. (if column is blank, use '-').

            For Column 38 , is Roll No.

            Column 39 (If present this column), is For 1st Division in Roman.
            Column 40 (If present this column), is For 2nd Division in Roman.

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.

            """

prompt_45_5 = """
            The image contains two rows:

            The first row shows system-generated numbers representing column names, ranging from Column 39 to Column 45.
            The second row contains handwritten data representing marks, subject names, or student names.
            Carefully check each column.

            Check data carefully vertically. The first row shows column numbers, and the second row contains handwritten data corresponding to each column.
            Ensure the data aligns exactly with the column number; do not shift any data between columns. Use vertical lines to help coordinate the column number
            with the data.

            Instructions:

            - Make sure the data corresponds to the correct column number.
            - If any data is missing or incorrect, do not shift the data from another column. Use `-` to indicate missing data.

            - For columns that require the sum of other columns, carefully check if the required columns are populated. If any of the summed columns are blank, use `-` in the sum.
            - For example, if you are summing columns (e.g., Column X + Column Y), ensure that you do not shift any data. If either column is blank, indicate this with `-` in the summed column.


            - If a column is blank, mark it with '-'.
            - Do not write '+' in the output; just write the final result in the corresponding column.
            - make sure the data is correponding to that column number.

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.
        Carefully check the data dont misplaced the column focus on out of marks and headline column number.

            For Column 37 (If present this column), Grand Total marks out of 1200. (if column is blank, use '-').

            For Column 38 (If present this column), is Roll No.

            Column 39 is For 1st Division in Roman.
            Column 40 is For 2nd Division in Roman.
            Column 41 is For 3rd Division in Roman.

            Column 42 , is for Distinction.

            Column 43, is for Fail.

            Column 44, is for For Marksheet No.

            Column 45, is for Remark.

            If the whole slice is blank marks it '-'

	    IMPORTANT = data we want in this format ONLY ----> "Column_No" - Data , EXAMPLE = Column_1 - 23,  dont write anything else.

            """


### global variables
Anchor1_cor=[]
Anchor2_cor=[]

tr_column_list=[]

error_result=0
total_image_length=0
stop=0

def Find_anchor_point(cordinates_point_path):
        global Anchor1_cor ,Anchor2_cor 
        def clean_and_format_data(data):
            cleaned_data = {}
            for variable_name, variable_data in data.items():
                cleaned_list = []
                for entry in variable_data:
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

        cleaned_data = clean_and_format_data(data_received)

        for key, value in cleaned_data.items():
            globals()[key] = value

            if key == 'Column_2':
                #print("inside anchor1")
                Anchor1_points = []
                Anchor1_points.extend(value)
                
                Anchor1_cor = Anchor1_points
                print("Anchor1_points", Anchor1_cor)
            if key == 'Column_29':
                Anchor2_points = []
                Anchor2_points.extend(value)
                
                Anchor2_cor = Anchor2_points
                print("Anchor2_points 29", Anchor2_cor)
            
            if key == 'Column_38':
                Anchor2_points = []
                Anchor2_points.extend(value)
                
                Anchor2_cor = Anchor2_points
                print("Anchor2_points 38", Anchor2_cor)

            if key == 'Column_39':
                Anchor2_points = []
                Anchor2_points.extend(value)
                Anchor2_cor = Anchor2_points
                print("Anchor2_points 39", Anchor2_cor)

            if key == 'Column_44':
                Anchor2_points = []
                Anchor2_points.extend(value)
                
                Anchor2_cor = Anchor2_points
                print("Anchor2_points 44", Anchor2_cor)
            
            if key == 'Column_47':
                Anchor2_points = []
                Anchor2_points.extend(value)
                
                Anchor2_cor = Anchor2_points
                print("Anchor2_points 47", Anchor2_cor)

            if key == 'Column_52':
                Anchor2_points = []
                Anchor2_points.extend(value)
                
                Anchor2_cor = Anchor2_points
                print("Anchor2_points 52", Anchor2_cor)
        
        return Anchor1_cor ,Anchor2_cor 



def anchor_parameter(Anchor1_cor,Anchor2_cor,yolo_model_1, yolo_model_2):
    Anchor1 = [value for coords in Anchor1_cor for value in coords]

    Anchor2 = [value for coords in Anchor2_cor for value in coords]
    crop_params = [
        {'model': yolo_model_1, 'x': Anchor1[0], 'y': Anchor1[1], 'width': Anchor1[2], 'height': Anchor1[3]},
        {'model': yolo_model_2, 'x': Anchor2[0], 'y': Anchor2[1], 'width': Anchor2[2], 'height': Anchor2[3]}
    ]
    return crop_params



class WorkerThread(QThread):
    finished = Signal(int)
    update_progress=Signal(int)
    update_progress_row=Signal(int)
    db_error=Signal(str)
    tr_remaining_time=Signal(str)
    tr_end_timer=Signal(str)
    gemini_error=Signal(str)
    tr_processing_error=Signal(str)
    def __init__(self, AppLauncher):
        super().__init__()  # Call the constructor of QThread
        self.AppLauncher = AppLauncher
        global Anchor1_cor, Anchor2_cor, cordinates_point_path,folder_path,college_folder_path ,output_path,anchor_cropped_path,rotate_path,table_name,db_file,\
            final_tr_column_list,tr_column,not_rotate_path,slices_path,row_slices_path,upper_image_path,tr_column
        global error_result,yolo_model_1,yolo_model_2,tr_column_user,cordinates_point_path,College_named_area,Header_folder,sub_slices_path

        folder_path = self.AppLauncher.input_folder_path.text()
        college_folder_path = self.AppLauncher.input_folder_path_college.text()
        print("college_folder_path ",college_folder_path)
        cordinates_point_path = self.AppLauncher.cordinate_path_anchor.text()
        base_folder_name = "output"
        suffix = 1
        while True:
            if suffix == 1:
                output_path = os.path.join(folder_path, base_folder_name)
            else:
                output_path = os.path.join(folder_path, f"{base_folder_name}{suffix}")
            
            if not os.path.exists(output_path):
                anchor_cropped_path = os.path.join(output_path, "Anchor Folder")
                os.makedirs(anchor_cropped_path, exist_ok=True)
                rotate_path = os.path.join(output_path, "Rotate Folder")
                os.makedirs(rotate_path, exist_ok=True)
                not_rotate_path = os.path.join(output_path, "Unmatch")
                os.makedirs(not_rotate_path, exist_ok=True)
                slices_path = os.path.join(output_path, "slices")
                os.makedirs(slices_path, exist_ok=True)

                sub_slices_path = os.path.join(output_path, "Sub_slices")
                os.makedirs(sub_slices_path, exist_ok=True)
                row_slices_path = os.path.join(output_path, "ROw Slices")
                os.makedirs(row_slices_path, exist_ok=True)
                upper_image_path= os.path.join(output_path, "Cropped Image")
                os.makedirs(upper_image_path, exist_ok=True)
                College_named_area= os.path.join(output_path, "College_named_area")
                os.makedirs(College_named_area, exist_ok=True)
                Header_folder= os.path.join(output_path, "Header")
                os.makedirs(Header_folder, exist_ok=True)
                break
            
            suffix += 1

        os.makedirs(output_path, exist_ok=True)

        table_name=self.AppLauncher.table_value.text()
        db_file=self.AppLauncher.db_path.text()
        tr_column= self.AppLauncher.columns_platform_row.currentText()
        tr_column_list.append(tr_column)
        final_tr_column_list=tr_column_list
        tr_column_user=int(tr_column_list[0])
        
        if tr_column_user==45:
            #tr_column=38
            tr_column=45
            #cordinates_point_path=cordinates_path_45
            yolo_model_1 = yolo_model_col45_1
            yolo_model_2 = yolo_model_col45_2
            #template_image=cv2.imread(header_45)
        elif tr_column_user==35:
            #tr_column=29
            tr_column=35
            #cordinates_point_path=cordinates_path_35
            yolo_model_1 = yolo_model_col45_1
            yolo_model_2 = yolo_model_col35_2
            #template_image=cv2.imread(header_35)
        elif tr_column_user==44:
            #tr_column=38
            tr_column=44
            #cordinates_point_path=cordinates_path_44
            yolo_model_1 =yolo_model_col44_1
            yolo_model_2 = yolo_model_col44_2
            #template_image=cv2.imread(header_44)

        elif tr_column_user==46:
            #tr_column=38
            tr_column=46
            #cordinates_point_path=cordinates_path_44
            yolo_model_1 =yolo_model_col46_1
            yolo_model_2 = yolo_model_col46_2
            #template_image=cv2.imread(header_46)
        elif tr_column_user==57:
            #tr_column=52
            tr_column=57
            #cordinates_point_path=cordinates_path_57
            yolo_model_1 =yolo_model_col57_1
            yolo_model_2 = yolo_model_col57_2 
            #template_image=cv2.imread(header_57)
        elif tr_column_user==50:
            #tr_column=44
            tr_column=50
            #cordinates_point_path=cordinates_path_50
            yolo_model_1 =yolo_model_col50_1
            yolo_model_2 = yolo_model_col50_2
            #template_image=cv2.imread(header_50)
        elif tr_column_user==52:
            #tr_column=47
            tr_column=52
            #cordinates_point_path=cordinates_path_52
            yolo_model_1 =yolo_model_col52_1
            yolo_model_2 = yolo_model_col52_2
            #template_image=cv2.imread(header_52)


    def run(self):
        global error_result,Anchor1_cor ,Anchor2_cor 
        Anchor1_cor ,Anchor2_cor =Find_anchor_point(cordinates_point_path)
        crop_params = anchor_parameter(Anchor1_cor,Anchor2_cor,yolo_model_1, yolo_model_2)
        start=time.time()
        self.process_tr(crop_params)
        end = time.time()
        total_time=end-start
        #total_time_hours = total_time / 3600
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_hours=f'{int(hours)} hr {int(minutes)} min {int(seconds)} sec'
        self.tr_end_timer.emit(total_time_hours)
        self.finished.emit(error_result)
    



    

    def process_tr(self,crop_params):
        h_iteration = 2
        v_iteration = 2
        h_rect = (20, 2)
        v_rect = (2, 20) 
        dil_h_rect = (25,1)
        dil_v_rect = (1,25)
        dil_v_iter = 1 
        dil_h_iter = 1
        
        def rotate(image, anchor1, anchor2):
            h, w = image.shape[:2]
            angle = math.degrees(math.atan2(anchor2[1] - anchor1[1], anchor2[0] - anchor1[0]))
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

# Apply the affine transformation (rotate the image)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))


            
            #rotated_image = imutils.rotate_bound(image, -angle)
            # if angle < 0 :
            #     rotated_image = imutils.rotate_bound(image, angle)
            # else:
            #     rotated_image = imutils.rotate_bound(image, -angle)
            # print("angle++++++++++++++++++++++++++++",angle)
            return rotated_image, angle
        
        def find_top_left_anchor(box):
            x1, y1, w, h, _, _ = box
            return int(x1), int(y1),int(w),int(h)
        
        def process_with_model_for_anchor(cropped_image, model):
            results = model.predict(source=cropped_image)
            max_conf_box = None
            max_confidence = 0
            for result in results:
                for box in result.boxes.data:
                    if box.shape[0] >= 5:
                        confidence = float(box[4].item())
                        if confidence > max_confidence:
                            max_confidence = confidence
                            max_conf_box = box.tolist()
            return find_top_left_anchor(max_conf_box) if max_conf_box is not None else None
        

            

        

        def get_centroids(image):
            #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image=image
            # Threshold the image
            _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

            # Extract horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, h_rect)
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations = h_iteration)

            dilate_horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dil_h_rect)
            horizontal_lines = cv2.dilate(horizontal_lines, dilate_horizontal_kernel, iterations= dil_h_iter)

            # Extract vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, v_rect)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations = v_iteration)

            # Dilate vertical lines to enhance them
            dilate_vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dil_v_rect)
            vertical_lines = cv2.dilate(vertical_lines, dilate_vertical_kernel, iterations=dil_v_iter)

            # Combine horizontal and vertical lines to get intersections
            intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

            # Find centroids of intersections
            contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
            return centroids
        

        def filter_adjacent_lines(rows):
            i = 0
            while i < len(rows) - 1:
                a = rows[i]
                b = rows[i + 1]
                if abs(a - b) < 20:
                    # Replace the first value with the median
                    rows[i] = int(np.median([a, b]))
                    # Remove the next element
                    del rows[i + 1]
                else:
                    # Move to the next pair
                    i += 1
            return rows
        

        def filter_adjacent_lines_column(rows):
            i = 0
            while i < len(rows) - 1:
                a = rows[i]
                b = rows[i + 1]
                if abs(a - b) < 40:
                    # Replace the first value with the median
                    rows[i] = int(np.median([a, b]))
                    # Remove the next element
                    del rows[i + 1]
                else:
                    # Move to the next pair
                    i += 1
            return rows


        def get_rows_n_cols(centroids):
            """
            Groups centroid coordinates into rows and columns based on proximity and filters them by a threshold.

            Args:
                centroids: A list of (x, y) centroid coordinates.
                row_lm: Minimum threshold for the number of points in a row.
                col_lm: Minimum threshold for the number of points in a column.

            Returns:
                A tuple of filtered rows and columns.
            """
            x_cords_grouped = group_consecutive_numbers(sorted(centroid[0] for centroid in centroids))
            y_cords_grouped = group_consecutive_numbers(sorted(centroid[1] for centroid in centroids))

            y_cords_grouped_1 = [(group) for group in y_cords_grouped if len(group) > 25]### change this for missing slice , original 25


            filtered_groups = []
            for group in y_cords_grouped_1:
                if len(group) >= 10:
                    filtered_groups.append((tuple(group[:5]), tuple(group[-5:])))

            medians = [ (int(np.median(group[0])), int(np.median(group[1])))  for group in filtered_groups]

            rows = [int(np.mean(points)) for points in medians]

            #rows = [int(np.median(lst)) for lst in y_cords_grouped]
            cols = [int(np.median(lst)) for lst in x_cords_grouped]

            filtered_rows = filter_adjacent_lines(rows)
            filtered_cols = filter_adjacent_lines_column(cols)

            # i = 0
            # while i < len(filtered_rows) - 1:
            #     if filtered_rows[i+1] - filtered_rows[i] < 50:  # Case 1: Difference is less than 50
            #         del filtered_rows[i+1]
            #     elif filtered_rows[i] - filtered_rows[i+1] < 50:  # Case 2: Reverse difference is less than 50
            #         del filtered_rows[i]
            #     else:
            #         i += 1  # Only increment if no deletion occurs



            return filtered_rows, filtered_cols

        def group_consecutive_numbers(numbers, delta=10):
            """
            Groups consecutive numbers into tuples based on a proximity threshold.

            Args:
                numbers: A list of integers.
                delta: Maximum difference allowed for grouping consecutive numbers.

            Returns:
                A list of tuples, where each tuple contains consecutive numbers.
            """
            if not numbers:
                return []

            grouped_numbers, current_group = [], [numbers[0]]  # Use list for efficient appending

            for num in numbers[1:]:
                if abs(num - current_group[-1]) <= delta:
                    current_group.append(num)  # Extend the current group
                else:
                    grouped_numbers.append(tuple(current_group))  # Save the group as a tuple
                    current_group = [num]  # Start a new group

            grouped_numbers.append(tuple(current_group))  # Add the last group
            return grouped_numbers

        def image_format(image):
            """Convert the image array to base64 encoding."""
            _, buffer = cv2.imencode('.png', image)
            img_byte = base64.b64encode(buffer).decode('utf-8')
            return [{"mime_type": "image/png", "data": img_byte}]


        # def parse_response(response_text):
        #     print("call parse response")
        #     """Parses the Gemini response into a list of data for insertion into the database."""
        #     parsed_data = ["-"] * tr_column  # Initialize with default values
        #     for line in response_text.strip().splitlines():
        #         if line.startswith("Column_"):
        #             try:
        #                 # Split into column identifier and value
        #                 col_num, value = line.split(" - ")
        #                 # Skip columns with invalid characters like '+'
        #                 if '+' in col_num:
        #                     print(f"Skipping line with invalid column name: '{col_num}'")
        #                     continue
        #                 # Extract the numeric part of the column index safely
        #                 col_num_split = col_num.split("_")[1]
        #                 if col_num_split.isdigit():  # Check if it's purely numeric
        #                     col_index = int(col_num_split) - 1
        #                     parsed_data[col_index] = value
        #             except Exception as e:
        #                 print(f"Error parsing line: '{line}' - {e}")
        #                 try:
        #                     col_num_split = col_num.split("_")[1]
        #                     if col_num_split.isdigit():
        #                         col_index = int(col_num_split) - 1
        #                         parsed_data[col_index] = "*"
        #                 except Exception:
        #                     print(f"Error assigning * for column: {col_num}")
        #     print("Parsed data:", parsed_data)
        #     return parsed_data
        def parse_response_printed(data):
            pairs = data.split('\n')
            result = {}
            try :


                for pair in pairs:
                    # Split only at the first occurrence of ' - ' or '-' to separate column name from value
                    key, value = re.split(r' - |-', pair, maxsplit=1)
                    result[key.strip()] = value.strip()
                return result
            except:
                return result
            
        def parse_response(response_text):
            #print("call parse response")
        
            # Initialize an empty dictionary to store column name-value pairs
            parsed_data = {}
            
            # Loop over each line in the response
            if response_text =="error":
                parsed_data=parsed_data
            else:
                for line in response_text.strip().splitlines():
                    if line.startswith("Column_") :
                        try:
                            # Split into column identifier and value
                            col_num, value = line.split(" - ")
                            
                            # Skip columns with invalid characters like '+'
                            if '+' in col_num:
                                print(f"Skipping line with invalid column name: '{col_num}'")
                                continue
                            
                            # Extract the numeric part of the column index safely
                            col_num_split = col_num.split("_")[1]
                            if col_num_split.isdigit():  # Ensure it's purely numeric
                                # Store the column name (e.g., "Column_1") and its corresponding value in the dictionary
                                parsed_data[col_num] = value
                        except Exception as e:
                            print(f"Error parsing line: '{line}' - {e}")
                            try:
                                # If parsing fails, store the column name with a placeholder value
                                col_num_split = col_num.split("_")[1]
                                if col_num_split.isdigit():
                                    parsed_data[col_num] = "*"
                            except Exception:
                                print(f"Error assigning '*' for column: {col_num}")
                    elif line.startswith("Column"):
                        try:
                            # Split into column identifier and value
                            col_num, value = line.split(" - ")
                            
                            # Skip columns with invalid characters like '+'
                            if '+' in col_num:
                                print(f"Skipping line with invalid column name: '{col_num}'")
                                continue
                            
                            # Extract the numeric part of the column index safely
                            col_num_split = col_num.split(" ")[1]
                            col_num=col_num.replace( " ","_")
                            if col_num_split.isdigit():  # Ensure it's purely numeric
                                # Store the column name (e.g., "Column_1") and its corresponding value in the dictionary
                                parsed_data[col_num] = value
                        except Exception as e:
                            print(f"Error parsing line: '{line}' - {e}")
                            try:
                                # If parsing fails, store the column name with a placeholder value
                                col_num_split = col_num.split(" ")[1]
                                if col_num_split.isdigit():
                                    parsed_data[col_num] = "*"
                            except Exception:
                                print(f"Error assigning '*' for column: {col_num}")
                    

            #print("Parsed data:", parsed_data)
            return parsed_data

                
        
        def gemini_prompt_digit(image_path,prompt):      
            """Generate a response from the Gemini model to extract digits from an image."""
            print("gemini callled")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image at {image_path} could not be read.")
            
            system_prompt = prompt
            image_info = image_format(image)
            input_prompt = [system_prompt] + image_info
            try:
                response = model.generate_content(input_prompt)  
                return response.text
            except  Exception as e33:
                #error_result=1
                #gemini_err=f"{e33}"
                #print("gemini_error e33",e33)
                #self.gemini_error.emit(gemini_err)
                response="error"
                return response
      

        def doctr_ocr(img_path):
            try:
                # Load the document
                doc = DocumentFile.from_images(img_path)
                
                # Load OCR model
                model = ocr_predictor(pretrained=True)

                # Perform OCR
                result = model(doc)

                # Extract text from doctr result
                extracted_text = []
                for block in result.export()['pages'][0]['blocks']:
                    for line in block['lines']:
                        line_text = " ".join(word['value'] for word in line['words'])
                        extracted_text.append(line_text)

                final_text = "\n".join(extracted_text)
                return final_text

            except Exception as e:
                # In case of any error, return "no data"
                print(f"Error during OCR processing: {e}")
                return "no data"

         
        def extract_college_name(image,image_name):

                # Perform object detection
                res = yolo_college_name.predict(source=image)
                boxes = res[0].boxes.xyxy.cpu().numpy() if len(res[0].boxes.xyxy) > 0 else []

                if len(boxes) > 0:  # Ensure at least one detection
                    # Get coordinates of the first detected object
                    x1, y1, x2, y2 = map(int, boxes[0])
                    
                    # Validate bounding box coordinates
                    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                        return None

                    # Crop the image
                    cropped_image = image[y1:y2, x1:x2]
                    if cropped_image is None or cropped_image.size == 0:
                        return None

                    save_path = os.path.join(College_named_area, f"{os.path.basename(image_name).split('.')[0]}_cropped.jpg")
                    cv2.imwrite(save_path, cropped_image)
                    prediction=doctr_ocr(save_path)
                    z=prediction.split("\n")
                    course_name=z[0]
                    college_name=" ".join(z[1:])
                    return course_name,college_name
                else:
                    print(f"No objects detected in")
                    course_name="not found"
                    college_name="not found"
                    return course_name,college_name



        def setup_database(db_file, table_name):
            try:   
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Generate column names from Column_1 to Column_(tr_column)
                columns = ["filename","slice_name","TR_Type","Row_Number","College_Name","Course_Name"] + [f"Column_{i} TEXT" for i in range(1, tr_column + 1)]
                columns_str = ", ".join(columns)
                
                # Create table if it doesn't exist
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})")
                conn.commit()           
                return conn, cursor
            except Exception as e1:
                error_result=1
                db_connection_error=f"{e1}"
                self.db_error.emit(db_connection_error)
                #self.close()  
            

        def insert_data(cursor, filename, table_name, slice_name, data,college_name,Course_Name,row_number):
            try:
                # Ensure tr_column is an integer representing the additional columns
                filename11=filename.split(".")[0]
                if not isinstance(tr_column, int):
                    raise ValueError("Expected tr_column to be an integer representing the number of additional columns.")
                
                # Ensure data length matches the number of expected columns
                expected_data_length = tr_column  # Columns start from Column_1 to Column_(tr_column)
                if len(data) != expected_data_length:
                    raise ValueError(f"Expected {expected_data_length} data elements, but got {len(data)}.")
                
                # Generate column names dynamically
                columns = ["filename","slice_name","TR_Type","Row_Number","College_Name","Course_Name"] + [f"Column_{i}" for i in range(1, tr_column + 1)]
                
                # Generate placeholders dynamically
                placeholders = ", ".join(["?" for _ in columns])
                
                # Prepare and execute the SQL query
                sql_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                cursor.execute(sql_query, [filename11,slice_name,tr_column,row_number,college_name,Course_Name]  + data)

            except Exception as e1:
                error_result=1
                db_connection_error=f"{e1}"
                self.db_error.emit(db_connection_error)
                #self.close()  

        def is_scientific_notation(number_str):
            # Regular expression to match scientific notation
            scientific_notation_pattern = re.compile(r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$')
            return bool(scientific_notation_pattern.match(number_str))
        
        def count_value(count):
                count+=1
                return count

        def processing_start(crop_params):
            try:
                global total_image_length,start_time
                batch_count=0
                start_time=0.0
                count=0
                image_filenames = [filename for filename in os.listdir(folder_path) if filename.lower().endswith((".jpg", ".jpeg", ".png"))]
                total_image_length=len(image_filenames)

                conn, cursor = setup_database(db_file,table_name)
                success_count = 0
                fail_count = 0

                for image_file in os.listdir(folder_path):
                    if stop == 1:
                        QMessageBox.showwarning("Warning", "Close button is pressed. Terminating batch processing.")
                        break
                    #print("filename", image_file)
                    image_path = os.path.join(folder_path, image_file)
                    #college_image_filename=f'{image_file.split("_")[0]}_upper.jpg'
                    college_image_filename=image_file
                    print("college_image_filename",college_image_filename)
                    image_path_college = os.path.join(college_folder_path, college_image_filename)
                    
                    # Check if the current path is a file and has a valid image extension
                    if not os.path.isfile(image_path):
                        print(f"Skipping {image_path}: Not a valid image file or is a directory.")
                        continue

                    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        #print(f"Skipping {image_path}: Not a valid image file format.")
                        continue

                    original_image = cv2.imread(image_path)
                    if original_image is None:
                        print(f"Failed to load image: {image_path}")
                        continue

                    h, w, _ = original_image.shape
                    ##########addition#######
                    # if w>slice_width:
                    #     slice_width=w
                    image_name = os.path.basename(image_path)
                    anchors = []
                    anchor_height=[]
                    batch_count=batch_count+1
                    self.update_progress.emit(batch_count)

                    total_image_left= total_image_length-batch_count
                    if count==5:
                        end_time = time.time()
                        total_time_taken=end_time-start_time
                        time_taken_one_image=total_time_taken/10 ##### one image
                        total_Processing_time_all_image=time_taken_one_image*(total_image_left)
                    
                    if count>5:
                        total_Processing_time_all_image=total_Processing_time_all_image-time_taken_one_image
                        remaining_time=total_Processing_time_all_image
                        hours, remainder = divmod(remaining_time, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        if is_scientific_notation(str(remaining_time)):

                            time_left=f'0 hr 0 min 0 sec'
                            self.tr_remaining_time.emit(time_left)
                        else:
                            time_left=f'{int(hours)} hr {int(minutes)} min {int(seconds)} sec'
                            self.tr_remaining_time.emit(time_left)
                        
                        if count==2:
                            count=0
                            #print("count value is zero",count)
                            time_left=f'wait.......'
                            self.tr_remaining_time.emit(time_left)
                            
                                
                    if count==0:
                        start_time=time.time()
                    count=count_value(count)
                    #print("count",count)
                    if os.path.isfile(image_path_college):
                        college_image=cv2.imread(image_path_college)
                        result1,result2=extract_college_name(college_image,image_name) 
                        Course_Name=result1
                        college_name=result2

                        if "-" in college_name:
                            college_name = college_name.replace("-", " ")
                    else:
                        Course_Name="not_found1"
                        college_name ="not_found1"




                    ### for anchor detection          
                    for params in crop_params:
                        x, y, crop_width, crop_height = params['x'], params['y'], params['width'], params['height']
                        model = params['model']

                        if x + crop_width > w or y + crop_height > h:
                            print(f"Skipping {image_path}: crop area exceeds image dimensions.")
                            continue

                        cropped_image = original_image[y:y + crop_height, x:x + crop_width]
                        cropped_path = os.path.join(anchor_cropped_path, f'{image_name}_{x}_{y}.jpg')  # Ensure file extension
                        cv2.imwrite(cropped_path, cropped_image)
                        #print(f"Cropped image saved to: {cropped_path}")
                        anchor = process_with_model_for_anchor(cropped_image, model)
                        print("anchor",anchor)
                        if anchor:
                            anchors.append((anchor[0] + x, anchor[1] + y))
                            anchor_height.append(anchor[3]-anchor[1])
                        else:
                            print("anchor not detected")
                            not_rotate_image_filename = os.path.join(not_rotate_path, f"{os.path.splitext(image_name)[0]}.jpg")
                            cv2.imwrite(not_rotate_image_filename, original_image)
                    if len(anchors) == 2:
                        print(111111111111111111111)
                        an1=anchors[0]
                        an2=anchors[1]
                        rotated_image, _ = rotate(original_image, anchors[0], anchors[1])
                        rot_h,rot_w= rotated_image.shape[:2]
                        cropped_image_filename = os.path.join(rotate_path, f"{os.path.splitext(image_name)[0]}_cropped.jpg")
                        cv2.imwrite(cropped_image_filename, rotated_image)
                        upper_x=0      
                        upper_y=an1[1]
                        final_cropped_image=rotated_image
                        final_cropped_height,final_cropped_width=final_cropped_image.shape[:2]
                        if final_cropped_height>=100 and final_cropped_width>=100:
                                UI_path12111=os.path.join(upper_image_path,f'org_image_{image_file}')
                                cv2.imwrite(UI_path12111,final_cropped_image)
                    
                                binary = cv2.adaptiveThreshold(final_cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 61, 21)
                                denoised = cv2.GaussianBlur(binary, (5, 5), 0)
                                centroids = get_centroids(denoised)
                                rows, cols = get_rows_n_cols(centroids)
                                
                                print("rows============",rows)
                                cols.sort()
                                filtered_data = [cols[0]]

                                for i in range(1, len(cols)):
                                    if cols[i] - filtered_data[-1] >= 45:
                                        filtered_data.append(cols[i])
                                cols=filtered_data
                                print("cols=====",cols)
                                

                                if tr_column_user==57:
                                    print("for tr 57)")
                                    cols_1=cols[:7] # 4 to 10
                                    print("cols_1",cols_1)
                                    cols_2=cols[6:14]  # 11 to 17
                                    print("cols_2",cols_2)
                                    cols_3=cols[13:27]  # 18 to 30
                                    print("cols_3",cols_3)
                                    cols_4=cols[26:37] # 31 to 40
                                    print("cols_4",cols_4)
                                    cols_5=cols[36:46] # 41 to 49
                                    print("cols_5",cols_5)
                                    col_6= cols[45:]
                                    cols_1_median=cols_1[-1]
                                    cols_2_median=cols_2[-1]
                                    cols_3_median=cols_3[-1]
                                    cols_4_median=cols_4[-1]
                                    cols_5_median=cols_5[-1]
                                        
                                    # print("cols_1_median",cols_1_median)
                                    # print("cols_2_median",cols_2_median)
                                    # print("cols_3_median",cols_3_median)
                                    # print("cols_4_median",cols_4_median)
                                    # print("cols_5_median",cols_5_median)
                                
                                elif tr_column_user==35:
                                    cols_1=cols[:6]
                                    cols_2=cols[5:13]
                                    cols_3=cols[12:20]
                                    cols_4=cols[19:27]
                                    cols_5=cols[26:]
                                    cols_1_median=cols_1[-1]
                                    cols_2_median=cols_2[-1]
                                    cols_3_median=cols_3[-1]
                                    cols_4_median=cols_4[-1]
                                    #cols_5_median=cols_5[-1]
                                    # print("cols_1_median",cols_1_median)
                                    # print("cols_2_median",cols_2_median)
                                    # print("cols_3_median",cols_3_median)
                                    # print("cols_4_median",cols_4_median)
                                    #print("cols_5_median",cols_5_median)
                                elif tr_column_user==46:
                                    print("tr16 column")
             

                                    cols_1=cols[:8]
                                    cols_2=cols[7:16]
                                    cols_3=cols[15:26]
                                    cols_4=cols[25:35]
                                    cols_5=cols[34:]
                                    cols_1_median=cols_1[-1]
                                    cols_2_median=cols_2[-1]
                                    cols_3_median=cols_3[-1]
                                    cols_4_median=cols_4[-1]
                                    #cols_5_median=cols_5[-1]
                                    # print("cols_1_median",cols_1_median)
                                    # print("cols_2_median",cols_2_median)
                                    # print("cols_3_median",cols_3_median)
                                    # print("cols_4_median",cols_4_median) 
                                    # 
                                    # 
                                elif tr_column_user==45:
                                    cols_1=cols[:8]  ## 4 to11
                                    cols_2=cols[7:17]  ## 12 to 20
                                    cols_3=cols[16:26] ## 21 to 29
                                    cols_4=cols[25:35]
                                    cols_5=cols[34:]
                                    cols_1_median=cols_1[-1] 
                                    cols_2_median=cols_2[-1]
                                    cols_3_median=cols_3[-1]
                                    cols_4_median=cols_4[-1]  
                                   
                                elif tr_column_user==52:
                                    print("tr 52 indexing")
                                    cols_1=cols[:8]
                                    cols_2=cols[7:19]
                                    cols_3=cols[18:31]
                                    cols_4=cols[30:42]
                                    cols_5=cols[41:]
                                    cols_1_median=cols_1[-1]
                                    cols_2_median=cols_2[-1]
                                    cols_3_median=cols_3[-1]
                                    cols_4_median=cols_4[-1]
                                else:
                                    cols_1=cols[:6]
                                    cols_2=cols[5:13]
                                    cols_3=cols[12:20]
                                    cols_4=cols[19:33]
                                    cols_5=cols[32:]

                                    cols_1_median=cols_1[-1]
                                    cols_2_median=cols_2[-1]
                                    cols_3_median=cols_3[-1]
                                    cols_4_median=cols_4[-1]

                                    # print("cols_1_median",cols_1_median)
                                    # print("cols_2_median",cols_2_median)
                                    # print("cols_3_median",cols_3_median)
                                    # print("cols_4_median",cols_4_median)

                                missing_slice_list=[]

                                if  rows[0] > 20 and  rows[0] < 100 or len(rows) == 30:
                                        loop_start_index=0
                                        new_header_image=final_cropped_image[5:rows[0]-5 , 0:final_cropped_width] 
                                        

                                else:
                                        loop_start_index = 1
                                        header_y1 = rows[0]
                                        header_y2 = rows[1]
                                        new_header_image = final_cropped_image[header_y1-2:header_y2-2, 0:final_cropped_width]
                                       
 
                                header_path=os.path.join(Header_folder, f"{os.path.splitext(image_file)[0]}_Header.jpg")
                                cv2.imwrite(header_path,new_header_image)
                
                                headline1=new_header_image 
                                

                                loop_count=0
                                name=os.path.basename(image_file).split(".jpg")[0]
                                for i in range(loop_start_index, len(rows)):  # Start from the second value to skip 31                                
                                    print("##########################################")
                                    print("i",i)
                                    loop_count=loop_count+1
                                    row_number=loop_count
                                    if loop_start_index==0:
                                        if i !=29:
                                            y1 = rows[i]
                                            y2 = rows[i + 1]
                                            if y2-y1<70:
                                                y1 = rows[i-1]
                                                y2 = y1+82
                                            
                                            if y2 <= y1 or y2 > final_cropped_height:
                                                print(f"Skipping invalid crop: y1={y1}, y2={y2}")
                                                continue

                                            # Crop the image using OpenCV (array slicing)
                                            # slice_image= final_cropped_image[y1-5:y2+5, 0:final_cropped_width]  # y1:y2 for height, 0:width for full width
                                            # #slice_image=cv2.resize(slice_image,())
                                            # slice_image_printed_data= final_cropped_image_b[y1-5:y2+5, 0:]  

                                                                                # Crop the image using OpenCV (array slicing)
                                            slice_image= final_cropped_image[y1-2:y2, 0:final_cropped_width]  # y1:y2 for height, 0:width for full width
                                           
                                        
                                        else:
                                            print( " i is 29")
                                            y1 = rows[i]
                                            slice_image= final_cropped_image[y1-2:, 0:final_cropped_width]  # y1:y2 for height, 0:width for full width 
                                            
                                    else:
                                        if i !=30:
                                            y1 = rows[i]
                                            y2 = rows[i + 1]
                                            if y2-y1<70:
                                                y1 = rows[i-1]
                                                y2 = y1+82
                                            
                                            if y2 <= y1 or y2 > final_cropped_height:
                                                print(f"Skipping invalid crop: y1={y1}, y2={y2}")
                                                continue

                                            # Crop the image using OpenCV (array slicing)
                                            # slice_image= final_cropped_image[y1-5:y2+5, 0:final_cropped_width]  # y1:y2 for height, 0:width for full width
                                            # #slice_image=cv2.resize(slice_image,())
                                            # slice_image_printed_data= final_cropped_image_b[y1-5:y2+5, 0:]  

                                                                                # Crop the image using OpenCV (array slicing)
                                            slice_image= final_cropped_image[y1-2:y2, 0:final_cropped_width]  # y1:y2 for height, 0:width for full width
                                           
                                        
                                        else:
                                            print( " i is 29")
                                            y1 = rows[i]
                                            slice_image= final_cropped_image[y1-2:, 0:final_cropped_width]  # y1:y2 for height, 0:width for full width 
                                        
                                    # slice_image=cv2.resize(slice_image,(headline1.shape[1],headline1.shape[0]))
                                    # slice_image_printed_data=cv2.resize(slice_image_printed_data,(headline2.shape[1],slice_image_printed_data.shape[0]))
                                    # headline_h, headline_w=headline1.shape[:2]
                                    # slice_image_h, slice_image_w=slice_image.shape[:2]
                                    # width_difference=abs(headline_w - slice_image_w)
                                    #print("width_difference",width_difference)
                                    print(" headline1 h,w",headline1.shape[:2])
                                    print("headline2 h,w",headline2.shape[:2])

                                    
                                    print("slice_image h,w",slice_image.shape[:2])
                                    print(" slice_image_printed_data h,w",slice_image_printed_data.shape[:2])

                                    
                                    slice_name = f'{name}_slice_{loop_count}.jpg'
                                    if slice_image.shape[0] > 0 and slice_image.shape[1] > 0:  # Ensure height and width > 0
                                        row_slice_output_path=os.path.join(row_slices_path, f"{os.path.splitext(image_file)[0]}_slice_{loop_count}.jpg")
                                        cv2.imwrite(row_slice_output_path, slice_image)
                                        row_slice_output_path_b=os.path.join(row_slices_path, f"{os.path.splitext(image_file)[0]}_first_three_col_slice_{loop_count}.jpg")
                                        cv2.imwrite(row_slice_output_path_b, slice_image_printed_data)

                                        # headline_resized = cv2.resize(headline1, (slice_image.shape[1], headline1.shape[0]))
                                        # combined_image = np.vstack((headline1, slice_image))# Stack the headline and slice vertically
                                        # bw_combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)# Convert to grayscale
                                        # output_path = os.path.join(slices_path, f"{os.path.splitext(image_file)[0]}_slice_{loop_count}.jpg")
                                        # cv2.imwrite(output_path, bw_combined_image)# Save the final black and white image

                                        # headline_resized2 = cv2.resize(headline2, (slice_image_printed_data.shape[1], headline2.shape[0]))
                                        # combined_image2 = np.vstack((headline2, slice_image_printed_data))# Stack the headline and slice vertically
                                        # bw_combined_image2b = cv2.cvtColor(combined_image2, cv2.COLOR_BGR2GRAY)# Convert to grayscale
                                        # output_path_b = os.path.join(slices_path, f"{os.path.splitext(image_file)[0]}_first_half_slice_{loop_count}.jpg")
                                        # cv2.imwrite(output_path_b, bw_combined_image2b)# Save the final black and white image
                                        
                                        overlap_pixels = 5  # Change as needed
                                        headline_resized = cv2.resize(headline1, (slice_image.shape[1], headline1.shape[0]))# Resize headline1 to match the slice width
                                        slice_image_cropped = slice_image[overlap_pixels:, :]# Remove overlap from slice_image
                                        combined_image = np.vstack((headline_resized, slice_image_cropped))# Stack the headline and adjusted slice vertically
                                        bw_combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)# Convert to grayscale
                                        output_path = os.path.join(slices_path, f"{os.path.splitext(image_file)[0]}_slice_{loop_count}.jpg")# Save final black & white image
                                        cv2.imwrite(output_path, bw_combined_image)

                                        # ====== SECOND IMAGE PROCESSING ======
                                        overlap_pixels1 = 10
                                        #Resize headline2 to match slice_image_printed_data width
                                        headline_resized2 = cv2.resize(headline2, (slice_image_printed_data.shape[1], headline2.shape[0]))

                                        # Remove overlap from slice_image_printed_data
                                        headline_resized2 = headline_resized2 [overlap_pixels1:, :]
                                        #slice_image_printed_cropped = slice_image_printed_data[overlap_pixels1:, :]

                                        # Stack the headline and adjusted slice vertically
                                        combined_image2 = np.vstack((headline_resized2, slice_image_printed_data))
                                        #combined_image2 = np.vstack((headline_resized2, slice_image_printed_cropped))

                                        # Convert to grayscale
                                        bw_combined_image2b = cv2.cvtColor(combined_image2, cv2.COLOR_BGR2GRAY)

                                        #Save final black & white image
                                        output_path_b = os.path.join(slices_path, f"{os.path.splitext(image_file)[0]}_first_half_slice_{loop_count}.jpg")
                                        cv2.imwrite(output_path_b, bw_combined_image2b)

                                        if tr_column_user==57:
                                            bw_combined_image1=bw_combined_image[:,:cols_1_median]
                                            bw_combined_image2=bw_combined_image[:,cols_1_median:cols_2_median]
                                            bw_combined_image3=bw_combined_image[:,cols_2_median:cols_3_median]
                                            #print("bw_combined_image3",bw_combined_image3)
                                            bw_combined_image4=bw_combined_image[:,cols_3_median:cols_4_median]
                                            bw_combined_image5=bw_combined_image[:,cols_4_median:cols_5_median]
                                            bw_combined_image6=bw_combined_image[:,cols_5_median:]

                                            output_path1 =os.path.join(sub_slices_path,f"{os.path.splitext(image_file)[0]}_sub_slice_1_{loop_count}.jpg")
                                            cv2.imwrite( output_path1 , bw_combined_image2b) 

                                            output_path2 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_2_{loop_count}.jpg")
                                            cv2.imwrite(output_path2, bw_combined_image1)
                                            output_path3 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_3_{loop_count}.jpg")
                                            cv2.imwrite(output_path3, bw_combined_image2)
                                            output_path4 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_4_{loop_count}.jpg")
                                            cv2.imwrite(output_path4, bw_combined_image3)
                                            output_path5 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_5_{loop_count}.jpg")
                                            cv2.imwrite(output_path5, bw_combined_image4)
                                            output_path6 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_6_{loop_count}.jpg")
                                            cv2.imwrite(output_path6, bw_combined_image5)
                                            output_path7 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_7_{loop_count}.jpg")
                                            cv2.imwrite(output_path7, bw_combined_image6)

                                        else:
                                            bw_combined_image1=bw_combined_image[:,:cols_1_median]
                                            bw_combined_image2=bw_combined_image[:,cols_1_median:cols_2_median]
                                            bw_combined_image3=bw_combined_image[:,cols_2_median:cols_3_median]
                                            #print("bw_combined_image3",bw_combined_image3)
                                            bw_combined_image4=bw_combined_image[:,cols_3_median:cols_4_median]
                                            bw_combined_image5=bw_combined_image[:,cols_4_median:]



                                            output_path1 =os.path.join(sub_slices_path,f"{os.path.splitext(image_file)[0]}_sub_slice_1_{loop_count}.jpg")
                                            cv2.imwrite( output_path1 , bw_combined_image2b) 

                                            output_path2 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_2_{loop_count}.jpg")
                                            cv2.imwrite(output_path2, bw_combined_image1)
                                            output_path3 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_3_{loop_count}.jpg")
                                            cv2.imwrite(output_path3, bw_combined_image2)
                                            output_path4 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_4_{loop_count}.jpg")
                                            cv2.imwrite(output_path4, bw_combined_image3)
                                            output_path5 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_5_{loop_count}.jpg")
                                            cv2.imwrite(output_path5, bw_combined_image4)
                                            output_path6 = os.path.join(sub_slices_path, f"{os.path.splitext(image_file)[0]}_sub_slice_6_{loop_count}.jpg")
                                            cv2.imwrite(output_path6, bw_combined_image5)
                                    
                                            

                                        # try:
                                        #     if tr_column_user==52:
                                                  
                                        #         prompt1=prompt_52_1
                                        #         prompt2=prompt_52_2 
                                        #         prompt3=prompt_52_3
                                        #         prompt4=prompt_52_4 
                                        #         prompt5=prompt_52_5 

                                        #     elif tr_column_user==35:
                                        #         prompt1=prompt_35_1
                                        #         prompt2=prompt_35_2 
                                        #         prompt3=prompt_35_3
                                        #         prompt4=prompt_35_4 
                                        #         prompt5=prompt_35_5 
                                        #     elif tr_column_user==44:
                                        #         prompt1=prompt_44_1
                                        #         prompt2=prompt_44_2 
                                        #         prompt3=prompt_44_3
                                        #         prompt4=prompt_44_4 
                                        #         prompt5=prompt_44_5 
                                        #     elif tr_column_user==45:
                                        #         prompt1=prompt_45_1
                                        #         prompt2=prompt_45_2 
                                        #         prompt3=prompt_45_3
                                        #         prompt4=prompt_45_4 
                                        #         prompt5=prompt_45_5

                                        #     elif tr_column_user==46:
                                        #         prompt1=prompt_46_1
                                        #         prompt2=prompt_46_2 
                                        #         prompt3=prompt_46_3
                                        #         prompt4=prompt_46_4 
                                        #         prompt5=prompt_46_5


                                        #     elif tr_column_user==50:
                                        #         prompt1=prompt_50_1
                                        #         prompt2=prompt_50_2 
                                        #         prompt3=prompt_50_3
                                        #         prompt4=prompt_50_4 
                                        #         prompt5=prompt_50_5 

                                        #     elif tr_column_user==57:
                                        #         prompt1=prompt_57_1
                                        #         prompt2=prompt_57_2 
                                        #         prompt3=prompt_57_3
                                        #         prompt4=prompt_57_4 
                                        #         prompt5=prompt_57_5 
                                        #         prompt6=prompt_57_6
                                 
                                        #     response_text0 = gemini_prompt_digit(output_path1,prompt_printed)
                                        #     print("response_text0",response_text0)
                                     
                                        #     parsed_data0 = parse_response_printed(response_text0)
                                        #     print("parsed_data0",parsed_data0) 


                                        #     response_text1 = gemini_prompt_digit(output_path2,prompt1)
                                        #     print("response_text1",response_text1)
                                        #     parsed_data1 = parse_response(response_text1)
                                        #     print("parsed_data1",parsed_data1)
                                    
                                        #     response_text2 = gemini_prompt_digit(output_path3,prompt2)
                                        #     #print("response_text2",response_text2)
                                        #     parsed_data2 = parse_response(response_text2)
                                        #     #print("parsed_data2",parsed_data2)

                                        #     response_text3 = gemini_prompt_digit(output_path4,prompt3)
                                        #    # print("response_text3",response_text3)
                                        #     parsed_data3 = parse_response(response_text3)
                                        #     #print("parsed_data3",parsed_data3)

                                            
                                        #     response_text4 = gemini_prompt_digit(output_path5,prompt4)
                                        #     #print("response_text4",response_text4)
                                        #     parsed_data4 = parse_response(response_text4)
                                        #     #print("parsed_data4",parsed_data4)

                                        #     response_text5 = gemini_prompt_digit(output_path6,prompt5)
                                        #     #print("response_text5",response_text5)
                                        #     parsed_data5 = parse_response(response_text5)
                                        #     #print("parsed_data5",parsed_data5)

                                        #     if tr_column_user==57:
                                        #         response_text6 = gemini_prompt_digit(output_path7,prompt6)
                                        #         parsed_data6 = parse_response(response_text6)
                                        #         #print("parsed_data6",parsed_data6)

                                                    
                                        #         parsed_data_dict = {key: value for d in [parsed_data0,parsed_data1, parsed_data2,parsed_data3,parsed_data4,parsed_data5,parsed_data6 ] for key, value in d.items()}
                                        #         parsed_data_dict = {}
                                        #         for d in [parsed_data0,parsed_data1, parsed_data2, parsed_data3, parsed_data4,parsed_data5,parsed_data6]:
                                        #             parsed_data_dict.update(d)

                                        #     else:
                                        #         parsed_data_dict = {key: value for d in [parsed_data0,parsed_data1, parsed_data2,parsed_data3,parsed_data4,parsed_data5 ] for key, value in d.items()}
                                        #         parsed_data_dict = {}
                                        #         for d in [parsed_data0,parsed_data1, parsed_data2, parsed_data3, parsed_data4,parsed_data5]:
                                        #             parsed_data_dict.update(d)

                                        #     #print("parsed_data2",parsed_data2)
                                        #     #print(" parsed_data_dict", parsed_data_dict)
                                        #     #print("###############################################################################")
                                        #     final_parsed_data_dict = {f'Column_{i}': '**' for i in range(1,tr_column+1 )}

                                        #     for key1, value1 in parsed_data_dict.items():
                                        #         if key1 in final_parsed_data_dict:  
                                        #             final_parsed_data_dict[key1] = value1  

                                        #     parsed_data = list(final_parsed_data_dict.values())
                                        #   #  print("parsed_data",parsed_data)
                                            
                                        #     #print("done parse data")
                                            
                                        #     insert_data(cursor, image_file,table_name,slice_name, parsed_data,college_name,Course_Name,row_number)
                                        #     missing_slice_list.append(slice_name)
                                        #     #print("done for slice number",slice_name)
                                        #     self.update_progress_row.emit(i)
                                        #     conn.commit()
                                        #     logging.info(f"Data from {image_file} successfully inserted into the database.")
                                        #     success_count += 1
                                        # except Exception as e:
                                        #     logging.error(f"Failed to process {image_file}: {e}")
                                        #     fail_count += 1

                                    else:
                                        print(f"Empty crop detected: y1={y1}, y2={y2}")
                                
                                print("missing_slice_list ",missing_slice_list)
                                if len(rows) != 30 :
                                    
                                    if len(missing_slice_list)!=30:
                                        slice_numbers = sorted(int(name.split('_slice_')[1].split('.')[0]) for name in missing_slice_list)
                                        print("slice_numbers",slice_numbers)
                                        #all_slices = set(range(min(slice_numbers), max(slice_numbers) + 2))
                                        all_slices = set(range(1,31))
                                        print("all_slices",all_slices)
                                        existing_slices = set(slice_numbers)
                                        print("existing_slices",existing_slices)
                                        missing_slices = all_slices - existing_slices  
                                        print("missing_slices",missing_slices)
                                        missing_files = [f'{name}_slice_{num}.jpg' for num in sorted(missing_slices)]
                                        print("missing_files",missing_files)

                                        
                                        for i in range(len(missing_files)):
                                            missing_parsed_data_dict = {f'Column_{i}': '**' for i in range(1, tr_column + 1)}
                                            missing_parsed_data = list(missing_parsed_data_dict.values())
                                            missing_slice_name = missing_files[i]
                                            missing_row_number = missing_slice_name.split('_slice_')[1].split('.')[0]
                                            self.update_progress_row.emit(missing_row_number)
                                            try:
                                                insert_data(cursor, image_file, table_name, missing_slice_name, missing_parsed_data, college_name, Course_Name, missing_row_number)
                                                conn.commit()
                                            except Exception as e:
                                                print(f"Error inserting data for {missing_slice_name}: {e}")

                   
                        else:
                                not_rotate_image_filename = os.path.join(not_rotate_path, f"{os.path.splitext(image_name)[0]}.jpg")
                                cv2.imwrite(not_rotate_image_filename, original_image)
                    else:

                        print(f"Could not find both anchors for {image_name}. Skipping rotation.")
                        not_rotate_image_filename = os.path.join(not_rotate_path, f"{os.path.splitext(image_name)[0]}.jpg")
                        cv2.imwrite(not_rotate_image_filename, original_image)
                conn.close()
                
                logging.info(f"Processing complete. Successfully processed {success_count} images. {fail_count} failed.")
            except Exception as e44:
                error_result=1
                tr_processing_error11=f"{e44}"
                self.tr_processing_error.emit(tr_processing_error11)
        processing_start(crop_params)


class AppLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.gui_layout()

    def gui_layout(self):
        
        self.setWindowTitle("TR Sheet")
        self.setGeometry(0, 0, 400, 100)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        name_label = QLabel("TR Reader", self)
        name_label.setStyleSheet("color:yellow; background-color:brown;")
        name_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        font = name_label.font()
        font.setPointSize(25)
        font.setBold(True)
        name_label.setFont(font)
        layout.addWidget(name_label)
        
        self.row_label = QLabel("Select TR Pattern(Column-Wise) ", self)
        layout.addWidget(self.row_label)
        self.list_widget_question = QListWidget()
        self.list_widget_question.addItems(["35","44","45","46", "50","52" ,"57"])
        self.list_widget_question.setSelectionMode(QListWidget.MultiSelection)
        self.list_widget_question.itemSelectionChanged.connect(self.update_row_options)
        layout.addWidget(self.list_widget_question)
        self.columns_platform_row = QComboBox()
        layout.addWidget(self.columns_platform_row)
        
        self.input_folder_path_label, input_folder_button,self.input_folder_path= self.add_input_folder(layout,"Input Folder-", "Browse", "Enter Image Folder Path")
        input_folder_button.setFixedSize(200, 20)
        self.input_folder_path.setFixedSize(400, 20)
        input_folder_button.clicked.connect(self.browse_input_folder)

        self.input_folder_college_label, input_folder_button_college,self.input_folder_path_college= self.add_input_folder2(layout,"Input Folder(college_area)-", "Browse", "Enter Image Folder Path")
        input_folder_button_college.setFixedSize(200, 20)
        self.input_folder_path_college.setFixedSize(400, 20)
        input_folder_button_college.clicked.connect(self.browse_input_folder2)
        
        self.db_label, db_button, self.db_path = self.add_input_group_db(layout, "SQLite Path -", "Browse", "Enter database path")
        db_button.setFixedSize(200, 20)
        self.db_path.setFixedSize(400, 20)
        db_button.clicked.connect(self.browse_db_path)

        table_layout = QHBoxLayout()
        self.table_label = QLabel("Table name -", self)
        table_layout.addWidget(self.table_label)
        self.table_value = QLineEdit(self)
        self.table_value.setPlaceholderText("Enter table name")
        table_layout.addWidget(self.table_value)
        layout.addLayout(table_layout)

        #self.template_label1 = QLabel("Anchor Cordinates-", self)
        #layout.addWidget(self.template_label1)
        self.submitbtn = QPushButton("Anchor Cordinates", self)
        self.submitbtn.clicked.connect(self.open_html_page_anchor)
        self.submitbtn.setStyleSheet("background-color: darkCyan; color: white;")
        layout.addWidget(self.submitbtn)
        self.cordinates_label, cordinate_button, self.cordinate_path_anchor = self.add_input_group_coordinates_anchor(layout, "Anchor Cordinates File -", "Browse", "Upload Anchor Cordinates File")
        cordinate_button.setFixedSize(200, 20)
        self.cordinate_path_anchor.setFixedSize(400, 20)
        cordinate_button.clicked.connect(self.browse_cordinate_path_anchor)
        

        
        self.progress_label = QLabel("Batch Processing Result", self)
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.BatchProcessing_label_path = QLabel(self)
        layout.addWidget(self.BatchProcessing_label_path)
        
        self.progress_label_row = QLabel("TR Row Processing Result", self)
        layout.addWidget(self.progress_label_row)
        self.progress_bar_row = QProgressBar(self)
        layout.addWidget(self.progress_bar_row)
        self.BatchProcessing_label_path_row = QLabel(self)
        layout.addWidget(self.BatchProcessing_label_path_row)
        
        time_layout = QHBoxLayout()
        self.All_image_time_label = QLabel("  Remaining  Time -", self)
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
        self.submitbtn.clicked.connect(self.thread_calling)  
        self.submitbtn.setStyleSheet("background-color: green; color: white;")

        closebtn = QPushButton("Close", self)
        closebtn.setFixedSize(200, 30)
        closebtn.clicked.connect(self.close_window)
        closebtn.setStyleSheet("background-color: red; color: white;")
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.submitbtn)
        button_layout.addWidget(closebtn)
        layout.addLayout(button_layout)

    def update_row_options(self):
        selected_items = [item.text() for item in self.list_widget_question.selectedItems()]
        self.columns_platform_row.clear()
        if selected_items:
            combined_text = ' , '.join(selected_items)
            # Limit the length of the combined text
            max_length = 100
            ellipsis = '...'
            if len(combined_text) > max_length:
                combined_text = combined_text[:max_length - len(ellipsis)] + ellipsis
            self.columns_platform_row.addItem(combined_text)  


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
        
    def browse_input_folder(self):
        filename = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.input_folder_path.setText(filename)

    def add_input_folder2(self, layout, label_text, button_text, placeholder_text):
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
        
    def browse_input_folder2(self):
        filename = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.input_folder_path_college.setText(filename)
    

    def add_input_group_coordinates_anchor(self, layout, label_text, button_text, placeholder_text):
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
    def browse_cordinate_path_anchor(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select Cordinates file', filter='cordinates file (*.txt)')
        self.cordinate_path_anchor.setText(filename)
    
    def open_html_page_anchor(self):
        print("open_html_page_anchor called")
        def open_html_file(file_path):
            print("file_path",file_path)
            print("open_html_file called")
            abs_path = os.path.abspath(file_path)
            print("abs_path",abs_path)
            url = f'file://{abs_path}'
            
            webbrowser.open(url)

        if __name__ == '__main__':
            open_html_file(html_path_anchor)
    

    
    def add_input_group_db(self, layout, label_text, button_text, placeholder_text):
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
    
    def browse_db_path(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select database File', filter='Image Files (*.db)')
        self.db_path.setText(filename)
    
    def close_window(self):
        global stop
        stop = 1
        self.close()


    
    def thread_calling(self):

        self.folder_path = self.input_folder_path.text()
        self.folder_path2 = self.input_folder_path_college.text()
        self.table_name = self.table_value.text()
        self.db_path_value= self.db_path.text()
        self.cordinates_point_path = self.cordinate_path_anchor.text()
        #self.tr_column= self.columns_platform_row.currentText()
        #self.template=self.template_path.text()
        try:
            # Check if all required fields are filled
            #if all([self.table_name,self.db_path_value,self.folder_path,self.cordinates_point_path,self.template ]):
            if all([self.table_name,self.db_path_value,self.folder_path,self.folder_path2,self.cordinates_point_path]):
                self.submitbtn.setDisabled(True)
                self.worker_thread = WorkerThread(self)
                self.worker_thread.start()
                self.worker_thread.finished.connect(self.result_done)
                self.worker_thread.tr_remaining_time.connect(self.tr_remaining_time)
                self.worker_thread.tr_end_timer.connect(self.tr_end_timer)
                self.worker_thread.update_progress.connect(self.batch_progress)
                self.worker_thread.update_progress_row.connect(self.batch_progress_row)
                self.worker_thread.db_error.connect(self.db_entry_error)
                self.worker_thread.gemini_error.connect(self.gemini_calling_error)
                self.worker_thread.tr_processing_error.connect(self.tr_processing_time_error)
                
            else:
                QMessageBox.information(self, "Input Error", "Please fill in all required fields.")
        except Exception as e:
            QMessageBox.information(self, "Error", f" Error: {e}")
            self.refresh()

        
   
    def tr_remaining_time(self,time_left):
        self.All_image_time_value.setText(f'{time_left}')

    def tr_end_timer(self,total_time_hours):
        self.end_time_value.setText(f'{total_time_hours}')    
        
    def db_entry_error(self,db_connection_error):    
            QMessageBox.information(self, "Error", f"Database Connection Error: {db_connection_error}" )
            self.refresh_tr()

    def gemini_calling_error(self,gemini_err):    
            QMessageBox.information(self, "Error", f"LLM Model Error: {gemini_err}" )
            self.refresh_tr()
    def tr_processing_time_error(self,tr_processing_error11):    
            QMessageBox.information(self, "Error", f"Error at Processing: {tr_processing_error11}" )
            self.refresh_tr()
    def refresh(self):
        global error_result
        error_result=0


    def refresh_tr(self):
        global total_image_length
        total_image_length=0
        self.refresh()
         
    def batch_progress(self, val):
        print("inside batch_progress")
        self.progress_bar.setMaximum(total_image_length)
        print("total_image_length ",total_image_length)
        self.progress_bar.setValue(val)
        print("val ",val)
        self.batch_processing_function(total_image_length,val)

    def batch_processing_function(self,total_image_length,val):
        print("batch_processing_function ")
        text = f"Processing Batch {val}/{total_image_length}"
        self.BatchProcessing_label_path.setText(text)

    def batch_progress_row(self, row_count):
        total_row=30
        self.progress_bar_row.setMaximum(total_row)
        self.progress_bar_row.setValue(row_count)
        self.batch_processing_function_row(total_row,row_count)

    def batch_processing_function_row(self,total_row,row_count):
        
        text = f"Processing Batch {row_count}/{total_row}"
        self.BatchProcessing_label_path_row.setText(text) 
         
    def result_done(self,error_result):
        print("inside result done")
        if error_result==0:
            QMessageBox.information(self, "Result", "OMR Result Prepared")
        else:
            QMessageBox.information(self, "Result", "OMR  Result Not Prepared")
        self.refresh_tr()
        self.submitbtn.setDisabled(False)

    

def process_gui():
    app = QApplication(sys.argv)
    window = AppLauncher()
    window.show()
    app.exec_()

if __name__ == "__main__":
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\BarcodeReaderMutex")
    last_error = ctypes.windll.kernel32.GetLastError()
    
    if last_error == 183:  # ERROR_ALREADY_EXISTS
        print("Another instance is already running.")
        sys.exit(1)
    
    process_gui()
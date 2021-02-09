# 모듈 불러오기
import tensorflow as tf
import cv2 as cv
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from jetbot import Camera
from jetbot import Robot

import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings



detect_fn = None
category_index = None

img = None

box_list = None
class_list = None
score_list = None

status_none = 0
status_center = 1
status_left = 2
status_right = 2

# Robot 객체 생성하기
robot = Robot()

def printText(img_data, x, y, str_text):
	font_color = (255, 255, 255)
	point_text = (x, y)
	cv.putText(img_data, str_text, point_text, cv.FONT_HERSHEY_PLAIN, 5, font_color, 10)

def load_model(strpath):
	start_time = time.time()
	detect_fn = tf.saved_model.load(strpath)
	end_time = time.time()
	print("load_model : Elapsed time : {:.2f}s" .format(end_time-start_time))
	return detect_fn

def load_cateogry(strpath):
	category_index = label_map_util.create_category_index_from_labelmap(strpath)
	return category_index

def evaluate_image(img_data):
	global box_list
	global class_list
	global score_list
	
	result_position = status_none
	
	# 데이터 전처리하기
	height, width, channels = img_data.shape
    
	tf_data = tf.convert_to_tensor(img_data)
	tf_reshape = tf_data[tf.newaxis, ...]

	start_time = time.time()
	result_detection = detect_fn(tf_reshape)
	end_time = time.time()
	print("detect_fn() Elapsed time : {:.2f}s" .format(end_time-start_time))

	box_list = result_detection["detection_boxes"][0].numpy()
	class_list = result_detection["detection_classes"][0].numpy().astype(np.int64)
	score_list = result_detection["detection_scores"][0].numpy()

	img_copy = np.copy(img_data)
	
	if score_list[0] > 0.5:
		top = int(box_list[0][0] * height)
		left = int(box_list[0][1] * width)
		bottom = int(box_list[0][2] * height)
		right = int(box_list[0][3] * width)
	    
		cv.rectangle(img_copy, (left, top), (right, bottom), (0, 0, 255), thickness=5)
		printText(img_copy, 30, 50, category_index[class_list[0]]['name'])
		
		center_pos = (left + right) // 2
		quantity = width // 8
		
		if center_pos > quantity  * 3 and center_pos < quantity * 5 :
			result_position = status_center
		elif center_pos < quantity  * 3:
			result_position = status_left
		elif center_pos > quantity * 5:
			result_position = status_right
			
	cv.imshow("TITLE", img_copy)
	
	return result_position
	
# JetBot 제어하기
def JetBotUp():
	robot.set_motors(0.1, 0.1)
	time.sleep(0.1)
	robot.set_motors(0.0, 0.0)
	robot.stop()

def JetBotDown():
	robot.set_motors(-0.1, -0.1)
	time.sleep(0.1)
	robot.set_motors(0.0, 0.0)
	robot.stop()

def JetBotLeft():
	robot.set_motors(0.1, -0.1)
	time.sleep(0.1)
	robot.set_motors(0.0, 0.0)
	robot.stop()

def JetBotRight():
	robot.set_motors(-0.1, 0.1)
	time.sleep(0.1)
	robot.set_motors(0.0, 0.0)
	robot.stop()
    
def JetBotStop():
	robot.set_motors(0.0, 0.0)
	robot.stop()

# GStreamer 파이프라인 얻기
def GetGstreamerPipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def main():
	global detect_fn
	global category_index
	
	global img

	# 변수 초기화하기
	str_model = "exported-models/my_centernet/saved_model"
	detect_fn = load_model(str_model)

	str_pbtxt = "annotations/label_map.pbtxt"
	category_index = load_cateogry(str_pbtxt)

	# 카메라 장치 얻기
	pipeline = GetGstreamerPipeline(640, 480, 640, 480, 10)

    # 비디오 장치 얻기
	cap = cv.VideoCapture(pipeline)

	while cap.isOpened():
		# 이미지 읽기
		result, img = cap.read()
		if not result:
			break

		# 이미지 출력하기
		result_pos = evaluate_image(img)
		if result_pos == status_center:
			JetBotUp()
			
		elif result_pos == status_left:
			JetBotLeft()
			
		elif result_pos == status_right:
			JetBotRight()

		else:
			JetBotStop()
			
		keydata = cv.waitKey(1)

		if keydata == ord('q'):
			break

	# 동영상 닫기
	if cap.isOpened():
		cap.release()
	
#
if __name__ == "__main__":
	main()
	
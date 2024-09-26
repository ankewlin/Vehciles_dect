import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
import time

# 记录开始时间
start_time = time.time()

# 加载YOLOv8模型（可选：选择不同大小的模型，如 'yolov8s.pt', 'yolov8m.pt'）
model = YOLO('yolov8s.pt')  # 使用YOLOv8的小模型

# 定义检测的类（只检测车辆相关的类别）
classes_to_detect = ['bicycle', 'car', 'motorbike', 'bus', 'truck']

# 打开视频文件
path = r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.103\20240501_20240501125647_20240501140806_125649.mp4'
filename = os.path.basename(path)
video_name = os.path.splitext(filename)[0]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

# 定义用于计算流量的线的位置
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_position = frame_height // 2  # 根据需要调整位置

# 初始化变量
frame_count = 0
output_data = []
test_frame_saved = False

# 车辆跟踪相关变量
vehicle_id_counter = 0  # 用于分配新的车辆ID
vehicles = {}  # 存储车辆信息，格式：{vehicle_id: {'box': [x1, y1, x2, y2], 'frames': n}}

# 定义输出间隔（以帧数为单位）
output_frame_interval = 5  # 每隔5帧(fps = 5)输出一次

# 定义一个函数来计算IOU（用于简单的目标跟踪）
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 使用YOLOv8模型进行推理
    results = model.predict(frame)

    # 提取检测结果
    detections = []
    for result in results[0].boxes:  # 只处理当前帧的检测
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # 获取检测框坐标
        class_id = int(result.cls[0])  # 获取检测到的类别ID
        confidence = result.conf[0]  # 获取置信度
        class_name = model.names[class_id]  # 获取类别名称
        if class_name in classes_to_detect:
            detections.append({'box': [x1, y1, x2, y2], 'class_id': class_id, 'confidence': confidence})

    # 车辆跟踪和速度计算
    current_vehicles = {}
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        matched = False
        for vehicle_id, vehicle_data in vehicles.items():
            prev_box = vehicle_data['box']
            iou = compute_iou([x1, y1, x2, y2], prev_box)
            if iou > 0.5:
                dx = x1 - prev_box[0]
                dy = y1 - prev_box[1]
                distance = np.sqrt(dx * dx + dy * dy)
                speed = distance * fps  # 速度计算，单位：像素/秒
                current_vehicles[vehicle_id] = {
                    'box': [x1, y1, x2, y2],
                    'speed': speed,
                    'frames': vehicle_data['frames'] + 1
                }
                matched = True
                break
        if not matched:
            vehicle_id_counter += 1
            current_vehicles[vehicle_id_counter] = {
                'box': [x1, y1, x2, y2],
                'speed': 0,
                'frames': 1
            }
    vehicles = current_vehicles

    # 计算密度：当前帧中的车辆数量
    density = len(detections)

    # 计算流量：判断位于指定线位置的车辆数量
    flow = 0
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        if y1 <= line_position <= y2:
            flow += 1

    # 计算平均速度
    speeds = [vehicle['speed'] for vehicle in vehicles.values() if vehicle['speed'] > 0]
    if speeds:
        average_speed = sum(speeds) / len(speeds)
    else:
        average_speed = 0

    # 每隔一定的帧数输出一次数据
    if frame_count % output_frame_interval == 0:
        output = {
            'Frame': frame_count,
            'Flow': flow,
            'Density': density,
            'Speed': average_speed
        }
        print(output)
        output_data.append(output)

cap.release()

# 保存结果到文件
df = pd.DataFrame(output_data)
df.to_csv(r"C:\Users\HEDY\Desktop\高速公路交通流数据\res\res_counts", index=False)

# 记录结束时间
end_time = time.time()

# 计算程序运行时间
execution_time = end_time - start_time

print(f"Program execution time: {execution_time} seconds")
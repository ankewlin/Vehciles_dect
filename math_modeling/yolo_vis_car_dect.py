import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
import time

# 记录开始时间
start_time = time.time()

# 加载YOLOv8模型
model = YOLO('yolov8s.pt')

# 定义检测的类（只检测车辆相关的类别）
classes_to_detect = ['bicycle', 'car', 'motorbike', 'bus', 'truck']

# 打开视频文件
path = r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.103\20240501_20240501140806_20240501152004_140807.mp4'
filename = os.path.basename(path)
video_name = os.path.splitext(filename)[0]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频输出设置
output_path = r"C:\Users\HEDY\Desktop\video_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 定义用于计算流量的线的位置
line_position = frame_height // 2  # 根据需要调整位置

# 初始化变量
frame_count = 0
output_data = []

# 定义输出间隔（以帧数为单位）
output_frame_interval = 5  # 每隔25帧(fps = 25)输出一次

# 定义视频处理的最大帧数（前1分钟的视频）
max_frames = int(fps * 60)


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
    if not ret or frame_count >= max_frames:  # 只处理前1分钟的帧数
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
            detections.append({'box': [x1, y1, x2, y2], 'class_name': class_name, 'confidence': confidence})

    # 可视化检测结果
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        class_name = detection['class_name']
        confidence = detection['confidence']

        # 画出检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 在检测框上显示类别名称和置信度
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 画出流量线
    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)

    # 保存输出帧到视频
    out.write(frame)

    # 每隔一定的帧数输出一次数据
    if frame_count % output_frame_interval == 0:
        # 计算流量：判断位于指定线位置的车辆数量
        flow = sum(1 for detection in detections if detection['box'][1] <= line_position <= detection['box'][3])

        # 输出结果
        output = {
            'Frame': frame_count,
            'Flow': flow,
            'Detections': len(detections)
        }
        print(output)
        output_data.append(output)

# 释放视频资源
cap.release()
out.release()

# 保存检测结果到CSV文件
df = pd.DataFrame(output_data)
df.to_csv(r"C:\Users\HEDY\Desktop\video_output_data.csv", index=False)

# 记录结束时间
end_time = time.time()
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds")
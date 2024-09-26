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
classes_to_detect = ['car', 'truck']

# 定义文件夹路径
folders = {
    '32.31.250.103': [
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.103\20240501_20240501125647_20240501140806_125649.mp4',
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.103\20240501_20240501140806_20240501152004_140807.mp4'
    ],
    '32.31.250.105': [
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.105\20240501_20240501115227_20240501130415_115227.mp4',
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.105\20240501_20240501130415_20240501141554_130415.mp4',
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.105\20240501_20240501141554_20240501152820_141555.mp4'
    ],
    '32.31.250.107': [
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.107\20240501_20240501114103_20240501135755_114103.mp4',
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.107\20240501_20240501135755_20240501161432_135755.mp4'
    ],
    '32.31.250.108': [
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.108\20240501_20240501113543_20240501135236_113542.mp4',
        r'C:\Users\HEDY\Desktop\高速公路交通流数据\32.31.250.108\20240501_20240501135236_20240501160912_135235.mp4'
    ]
}

# 定义输出文件夹
output_folder = r'C:\Users\HEDY\Desktop\output_folder'
os.makedirs(output_folder, exist_ok=True)

# 定义输出间隔（以帧数为单位）
output_frame_interval = 5  # 每隔5帧输出一次数据

# 遍历文件夹中的每个视频文件
for folder, video_paths in folders.items():
    for path in video_paths:
        filename = os.path.basename(path)
        output_video_path = os.path.join(output_folder, f"output_{filename}")
        output_data = []

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定义视频输出设置
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # 定义用于计算流量的线的位置
        line_position = frame_height // 2  # 根据需要调整位置

        # 初始化变量
        frame_count = 0
        vehicles = {}
        vehicle_id_counter = 0
        car_speeds = {}
        truck_speeds = {}


        # 计算IOU的函数
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
            if not ret:  # 如果没有读取到帧，结束循环
                break
            frame_count += 1

            # 使用YOLOv8模型进行推理
            results = model.predict(frame)

            # 提取检测结果
            detections = []
            for i, result in enumerate(results[0].boxes):  # 只处理当前帧的检测
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # 获取检测框坐标
                class_id = int(result.cls[0])  # 获取检测到的类别ID
                confidence = result.conf[0]  # 获取置信度
                class_name = model.names[class_id]  # 获取类别名称
                if class_name in classes_to_detect:
                    detections.append({'box': [x1, y1, x2, y2], 'class_name': class_name, 'confidence': confidence})

            # 车辆跟踪和速度计算
            current_vehicles = {}
            for detection in detections:
                x1, y1, x2, y2 = detection['box']
                class_name = detection['class_name']
                matched = False
                for vehicle_id, vehicle_data in vehicles.items():
                    prev_box = vehicle_data['box']
                    iou = compute_iou([x1, y1, x2, y2], prev_box)
                    if iou > 0.5:
                        # 如果匹配，计算速度
                        dx = x1 - prev_box[0]
                        dy = y1 - prev_box[1]
                        distance = np.sqrt(dx * dx + dy * dy)
                        speed = distance * fps  # 速度计算，单位：像素/秒
                        current_vehicles[vehicle_id] = {
                            'box': [x1, y1, x2, y2],
                            'speed': speed,
                            'frames': vehicle_data['frames'] + 1,
                            'class_name': class_name
                        }

                        # 根据车辆类型分类记录速度
                        if class_name == 'car':
                            car_speeds[vehicle_id] = speed
                        elif class_name == 'truck':
                            truck_speeds[vehicle_id] = speed

                        matched = True
                        break
                if not matched:
                    # 如果没有匹配到，则为该检测分配新的ID
                    vehicle_id = vehicle_id_counter
                    vehicle_id_counter += 1
                    current_vehicles[vehicle_id] = {
                        'box': [x1, y1, x2, y2],
                        'speed': 0,  # 初始速度为0
                        'frames': 1,
                        'class_name': class_name
                    }

                    # 新车辆初始速度记录
                    if class_name == 'car':
                        car_speeds[vehicle_id] = 0
                    elif class_name == 'truck':
                        truck_speeds[vehicle_id] = 0

            # 更新车辆数据
            vehicles.update(current_vehicles)

            # 可视化检测结果并显示速度
            for vehicle_id, vehicle_data in current_vehicles.items():
                x1, y1, x2, y2 = vehicle_data['box']
                speed = vehicle_data['speed']
                class_name = vehicle_data['class_name']
                color = (0, 255, 0) if class_name == 'car' else (255, 0, 0)  # 绿色表示车，蓝色表示卡车
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {speed:.2f} px/s"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 画出流量线
            cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)

            # 保存输出帧到视频
            out.write(frame)

            # 每隔一定的帧数输出一次数据
            if frame_count % output_frame_interval == 0:
                # 输出每辆车/卡车的速度数据
                output = {'Frame': frame_count}
                for vehicle_id in sorted(car_speeds.keys()):
                    output[f"car{vehicle_id}"] = car_speeds[vehicle_id]
                for vehicle_id in sorted(truck_speeds.keys()):
                    output[f"truck{vehicle_id}"] = truck_speeds[vehicle_id]
                output_data.append(output)

        # 释放视频资源
        cap.release()
        out.release()

        # 转置数据以符合输出格式
        transposed_data = {}
        for d in output_data:
            for k, v in d.items():
                if k not in transposed_data:
                    transposed_data[k] = []
                transposed_data[k].append(v)

        # 保存车辆速度数据到CSV，按car1, car2, ..., truck1, truck2, ...格式
        df = pd.DataFrame(transposed_data)
        df.to_csv(os.path.join(output_folder, f"speeds_{filename}.csv"), index=False)

# 记录结束时间
end_time = time.time()
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds")

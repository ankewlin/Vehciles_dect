import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
import time

# 记录开始时间
start_time = time.time()

output_data = [114521]
df = pd.DataFrame(output_data)
df.to_csv(r"C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\res\5video11.csv", index=False)

# 记录结束时间
end_time = time.time()

# 计算程序运行时间
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds")
import cv2

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频时长（秒）
    duration = total_frames / fps

    print(f"视频总帧数: {total_frames}")
    print(f"视频帧率: {fps}")
    print(f"视频时长（秒）: {duration}")

    cap.release()

video_path_11= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.103\20240501_20240501125647_20240501140806_125649.mp4'
video_path_12= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.103\20240501_20240501140806_20240501152004_140807.mp4'

video_path_21= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.105\20240501_20240501115227_20240501130415_115227.mp4'
video_path_22= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.105\20240501_20240501130415_20240501141554_130415.mp4'
video_path_23= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.105\20240501_20240501141554_20240501152820_141555.mp4'

video_path_31= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.107\20240501_20240501114103_20240501135755_114103.mp4'
video_path_32= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.107\20240501_20240501135755_20240501161432_135755.mp4'

video_path_41= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.108\20240501_20240501113543_20240501135236_113542.mp4'
video_path_42= r'C:\Users\HEDY\Desktop\2.2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\2024年中国研究生数学建模竞赛E题数据\高速公路交通流数据\32.31.250.108\20240501_20240501135236_20240501160912_135235.mp4'

print(get_video_info(video_path=video_path_11))
print(get_video_info(video_path=video_path_12))

print(get_video_info(video_path=video_path_21))
print(get_video_info(video_path=video_path_22))
print(get_video_info(video_path=video_path_23))

print(get_video_info(video_path=video_path_31))
print(get_video_info(video_path=video_path_32))

print(get_video_info(video_path=video_path_41))
print(get_video_info(video_path=video_path_42))
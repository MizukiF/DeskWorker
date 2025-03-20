import cv2
import depthai as dai
import blobconverter
import numpy as np
import time
from collections import deque

is_deskworking = False
start_time = None
break_time = None
ref = deque(maxlen=300)
flag_thresthold = 100  # 'flag_threshold' の名前を統一

# フレームの正規化
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# 顔サイズが閾値以上かどうか判定する関数
def bbsize(frame, xmin, ymin, xmax, ymax):
    xlen = xmax - xmin
    ylen = ymax - ymin
    xratio = xlen / frame.shape[1]
    yratio = ylen / frame.shape[0]
    
    # サイズ閾値を適切に調整する
    return xratio > 0.1 and yratio > 0.15  # 例: 少し小さな顔も検出するように変更

# 判定フラグの更新とチェック
def check_flag(ref):
    if len(ref) < flag_thresthold:
        return None  # 判定できるデータが足りない場合は何もしない

    last_flags = list(ref)[-flag_thresthold:]  # 直近の判定データ

    if all(last_flags):  # True が連続
        return "Start"
    elif not any(last_flags):  # False が連続
        return "Break"
    return None

# Pipeline の作成
pipeline = dai.Pipeline()

# カメラノード（RGBカメラ）
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(300, 300)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# 顔検出用の ImageManip ノード
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(300, 300)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
cam.preview.link(manip.inputImage)

# MobileNetSSD 顔検出ネットワーク
face_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_nn.setConfidenceThreshold(0.5)
face_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
manip.out.link(face_nn.input)

# 出力ノード（カメラ画像 & 検出結果）
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam.preview.link(xout_video.input)

xout_det = pipeline.create(dai.node.XLinkOut)
xout_det.setStreamName("detections")
face_nn.out.link(xout_det.input)

# デバイスとパイプラインの開始
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video", maxSize=4, blocking=False)
    q_det = device.getOutputQueue("detections", maxSize=4, blocking=False)

    while True:
        frame = q_video.get().getCvFrame()
        detections = q_det.tryGet()
        
        detected = False

        if detections is not None:
            for detection in detections.detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                if bbsize(frame, bbox[0], bbox[1], bbox[2], bbox[3]):
                    detected = True
                    break
                
        ref.append(detected)
        flag = check_flag(ref)
    
        if flag == "Start" and not is_deskworking:
            start_time = time.time()
            is_deskworking = True
            print(f"Deskwork started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        elif flag == "Break" and is_deskworking:
            break_time = time.time()
            is_deskworking = False
            print(f"Deskwork breaked at {time.strftime('%H:%M:%S', time.localtime(break_time))}")

        # 画像を表示
        cv2.imshow("OAK-D Lite Face Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

import cv2
import depthai as dai
import blobconverter
import numpy as np
import time
from datetime import datetime
from collections import deque

is_deskworking = False
start_time = None
break_time = None
ref = deque(maxlen=5)
flag_thresthold = 3


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def bbsize(frame, xmin, ymin, xmax, ymax):

    # バウンディングボックスの幅と高さを計算
    xlen = xmax - xmin
    ylen = ymax - ymin
    xratio = xlen / frame.shape[1]
    yratio = ylen / frame.shape[0]
    
    # デバッグ用の出力
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Bounding box (normalized): {bbox}")
    print(f"Bounding box (pixels): ({xmin}, {ymin}) to ({xmax}, {ymax})")
    print(f"Width ratio: {xratio}, Height ratio: {yratio}")
    
    # 面積の比率で判定
    if xratio > 0.2 and yratio > 0.3:
        return True
    else:
        return False

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(1080, 1080)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

cam_xout = pipeline.create(dai.node.XLinkOut)
cam_xout.setStreamName("color")
cam.preview.link(cam_xout.input)

face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
cam.preview.link(face_det_manip.inputImage)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Spatial Detection network
face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
face_det_nn.setBoundingBoxScaleFactor(0.8)
face_det_nn.setDepthLowerThreshold(100)
face_det_nn.setDepthUpperThreshold(5000)
stereo.depth.link(face_det_nn.inputDepth)

face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
face_det_manip.out.link(face_det_nn.input)

face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

with dai.Device(pipeline) as device:
    device.startPipeline()
    q_color = device.getOutputQueue("color", maxSize=4, blocking=False)
    q_detection = device.getOutputQueue("detection", maxSize=4, blocking=False)
    
    while True:
        in_color = q_color.tryGet()
        in_detection = q_detection.tryGet()
        
        if in_color is not None:
            frame = in_color.getCvFrame()
            
        if in_detection is not None:
            detections  = in_detection.detections
            
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                z_coord = detection.spatialCoordinates.z / 1000
                
                if bbsize(frame,bbox[0],bbox[1],bbox[2],bbox[3]):
                    print("Detected")
                    ref.append(True)
                    
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)
          
        if cv2.waitKey(2000) == ord('q'):
            break
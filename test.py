from pathlib import Path

import cv2
import depthai as dai
import blobconverter
import numpy as np

pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd",shaves=6))

detection_nn.setConfidenceThreshold(0.5)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

cam_rgb.preview.link(xout_rgb.input)
cam_rgb.preview.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    
    frame = None
    detections = []
    
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1)* normVals).astype(int)
    
    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            
        if in_nn is not None:
            detections = in_nn.detections
            
        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                
            cv2.imshow("preview", frame)
        
        if cv2.waitKey(1) == ord("q"):
            break
                
        

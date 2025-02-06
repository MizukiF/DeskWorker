import cv2
import depthai as dai
import blobconverter
import numpy as np

pipeline = dai.Pipeline()

# Load models 

# Detection model
detection_model_path = blobconverter.from_zoo(
    name="face-detection-retail-0005",
    zoo_type="intel",
    shaves=6
)

# Pose estimation model
pose_model_path = blobconverter.from_zoo(
    name="head-pose-estimation-adas-0001",
    zoo_type="intel",
    shaves=6
)

MIN_THRESHOLD = 15 # yaw/pitch/roll threshold

# Add camera node
cam = pipeline.createColorCamera()
cam.setPreviewSize(600, 600)
cam.setInterleaved(False)
cam.setFps(30)

# Add Face detection model for detection
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(detection_model_path)

# Resize for face detection
face_manip = pipeline.createImageManip()
face_manip.initialConfig.setResize(300, 300)
face_manip.initialConfig.setKeepAspectRatio(True)


# Output nodes
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

xout_face = pipeline.createXLinkOut()
xout_face.setStreamName("face_detection")

xout_pose = pipeline.createXLinkOut()
xout_pose.setStreamName("pose_estimation")

# Create link
cam.preview.link(face_manip.inputImage)
face_manip.out.link(detection_nn.input)
detection_nn.out.link(xout_face.input)

# Run
with dai.Device(pipeline) as device:
    video_q = device.getOutputQueue("rgb", maxSize = 4, blocking = False)
    face_q = device.getOutputQueue("face_detection", maxSize = 4, blocking = False)
    
    while True:
        in_video = video_q.get()
        frame = in_video.getCvFrame()
        
        # Results of face detection (Bounding box dipiction)
        in_face = face_q.tryGet()
        if in_face is not None:
            detections = in_face.getFirstLayerFp16()
            for i in range(0, len(detections), 7):  # 7つごとに分割（モデルの出力フォーマット）
                conf = detections[i+2]  # 信頼度 (confidence score)
                if conf > 0.5:  # 閾値（しきい値）を設定
                    x1 = int(detections[i+3] * frame.shape[1])
                    y1 = int(detections[i+4] * frame.shape[0])
                    x2 = int(detections[i+5] * frame.shape[1])
                    y2 = int(detections[i+6] * frame.shape[0])
                
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            
        # Play video
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
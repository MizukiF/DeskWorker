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
detection_nn.setConfidenceThreshold(0.5)

# Resize for face detection
face_manip = pipeline.createImageManip()
face_manip.initialConfig.setResize(300, 300)
face_manip.initialConfig.setKeepAspectRatio(True)


# Add Pose estimation model
pose_nn = pipeline.createNeuralNetwork()
pose_nn.setBlobPath(pose_model_path)

# Resize for pose estimation
pose_manip = pipeline.createImageManip()
pose_manip.setWaitForConfigInput(True)
pose_manip.initialConfig.setResize(60, 60)
pose_manip.initialConfig.setKeepAspectRatio(True)

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

detection_nn.out.link(pose_manip.inputConfig) # Bounding box information
face_manip.out.link(pose_manip.inputImage) # Frame image
pose_manip.out.link(pose_nn.input) 

pose_nn.out.link(xout_pose.input)

# Run
with dai.Device(pipeline) as device:
    video_q = device.getOutputQueue("rgb", maxSize = 4, blocking = False)
    face_q = device.getOutputQueue("face_detection", maxSize = 4, blocking = False)
    pose_q = device.getOutputQueue("pose_estimation", maxSize = 4, blocking = False)
    
    while True:
        in_video = video_q.get()
        frame = in_video.getCvFrame()
        
        # Results of face detection (Bounding box dipiction)
        in_face = face_q.tryGet()
        if in_face is not None:
            for detection in in_face.detections:
                x1, y1, x2, y2 = (
                    int(detection.xmin * frame.shape[1]),
                    int(detection.ymin * frame.shape[0]),
                    int(detection.xmax * frame.shape[1]),
                    int(detection.ymax * frame.shape[0])
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        # Results of pose estimation
        in_pose = pose_q.tryGet()
        if in_pose is not None:
            yaw = in_pose.getFirstLayerFp16("angle_y_fc")[0]
            pitch = in_pose.getFirstLayerFp16("angle_p_fc")[0]
            roll = in_pose.getFirstLayerFp16("angle_r_fc")[0]

            
            vals = np.array([abs(pitch),abs(yaw),abs(roll)])
            max_index = np.argmax(vals)
            
            if vals[max_index] < MIN_THRESHOLD:
                txt = "Face Forward"
            elif max_index == 0:
                if pitch > 0: txt = "Look down"
                else: txt = "Look up"
            elif max_index == 1:
                if yaw >0: txt = "Turn right"
                else: txt = "Turn left"
            elif max_index == 2:
                if roll > 0: txt = "Tilt right"
                else: txt = "Tilt left"
            
            cv2.putText(frame, f"{txt} (Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f})",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
        # Play video
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
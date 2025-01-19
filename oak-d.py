import cv2
import depthai as dai
import blobconverter
import numpy as np

# Create a pipeline
pipeline = dai.Pipeline()

# Load models
detection_model_path = blobconverter.from_zoo(
    name="yolov7tiny_coco_416x416",
    zoo_type="depthai",
    shaves=6
)

postureEstimation_model_path = "facial-landmarks-35-adas-0002.blob"

print(f"detection model file used: {detection_model_path}")
print(f"posture estimation model file used: {postureEstimation_model_path}")

# Add camera node
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 640)
cam.setInterleaved(False)
cam.setFps(30)

# YOLO model for detection
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(detection_model_path)

# Resize for YOLO
yolo_manip = pipeline.createImageManip()
yolo_manip.initialConfig.setResize(416, 416)
yolo_manip.initialConfig.setKeepAspectRatio(True)
cam.preview.link(yolo_manip.inputImage)
yolo_manip.out.link(detection_nn.input)

# Facial landmarks model
landmarks_nn = pipeline.createNeuralNetwork()
landmarks_nn.setBlobPath(postureEstimation_model_path)

# Resize for facial landmarks
landmarks_manip = pipeline.createImageManip()
landmarks_manip.initialConfig.setResize(160, 160)
landmarks_manip.initialConfig.setKeepAspectRatio(True)

# Output nodes
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

xout_yolo = pipeline.createXLinkOut()
xout_yolo.setStreamName("yolo")
detection_nn.out.link(xout_yolo.input)

xout_landmarks = pipeline.createXLinkOut()
xout_landmarks.setStreamName("landmarks")
landmarks_nn.out.link(xout_landmarks.input)

# Run the pipeline
with dai.Device(pipeline) as device:
    # Queues
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    yolo_queue = device.getOutputQueue(name="yolo", maxSize=4, blocking=False)
    landmarks_queue = device.getOutputQueue(name="landmarks", maxSize=4, blocking=False)

    while True:
        # Get video frame
        rgb_frame = rgb_queue.get().getCvFrame()
        frame_height, frame_width = rgb_frame.shape[:2]

        # Get YOLO detection results
        yolo_data = yolo_queue.tryGet()
        if yolo_data:
            raw_data = yolo_data.getFirstLayerFp16()
            detections = []

            # YOLO output decoding
            if raw_data and len(raw_data) >= 7 and len(raw_data) % 7 == 0:
                for i in range(0, len(raw_data), 7):
                    detection = {
                        "label": int(raw_data[i + 1]),
                        "confidence": raw_data[i + 2],
                        "xmin": raw_data[i + 3],
                        "ymin": raw_data[i + 4],
                        "xmax": raw_data[i + 5],
                        "ymax": raw_data[i + 6]
                    }
                    if detection["confidence"] > 0.5:
                        detections.append(detection)
            else:
                print("Unexpected YOLO output format or no detections.")

            for detection in detections:
                if detection["label"] == 0:  # Person class in COCO dataset
                    x1 = int(detection["xmin"] * frame_width)
                    y1 = int(detection["ymin"] * frame_height)
                    x2 = int(detection["xmax"] * frame_width)
                    y2 = int(detection["ymax"] * frame_height)

                    # Draw YOLO bounding box
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Prepare landmarks input
                    landmarks_manip_cfg = landmarks_manip.initialConfig.get()
                    landmarks_manip_cfg.setCropRect(
                        detection["xmin"], detection["ymin"],
                        detection["xmax"], detection["ymax"]
                    )
                    landmarks_manip.initialConfig.setResize(160, 160)
                    landmarks_manip.inputImage.setConfig(landmarks_manip_cfg)
                    landmarks_manip.out.link(landmarks_nn.input)

        # Get landmarks results
        landmarks_data = landmarks_queue.tryGet()
        if landmarks_data:
            landmarks = landmarks_data.getFirstLayerFp16()
            for i in range(0, len(landmarks), 2):
                x = int(landmarks[i] * frame_width)
                y = int(landmarks[i + 1] * frame_height)
                cv2.circle(rgb_frame, (x, y), 2, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow("Camera with Landmarks", rgb_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
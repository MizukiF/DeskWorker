import cv2
import depthai as dai
import blobconverter
import numpy as np

pipeline = dai.Pipeline()

# Load models 
detection_model_path = blobconverter.from_zoo(
    name="face-detection-retail-0005",
    zoo_type="intel",
    shaves=6
)

pose_model_path = blobconverter.from_zoo(
    name="head-pose-estimation-adas-0001",
    zoo_type="intel",
    shaves=6
)

MIN_THRESHOLD = 15  # yaw/pitch/roll threshold

# 🎥 1️⃣ カメラノード
cam = pipeline.createColorCamera()
cam.setPreviewSize(300, 300)  # 🔥 600×600 → 300×300 に修正
cam.setInterleaved(False)
cam.setFps(30)

# 🏷️ 2️⃣ 顔検出モデル
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(detection_model_path)

# 🔳 3️⃣ ImageManip - 顔のクロップとリサイズ (60x60)
pose_manip = pipeline.createImageManip()
pose_manip.inputConfig.setWaitForMessage(True)  # 🔥 修正
pose_manip.initialConfig.setKeepAspectRatio(True)  # 🔥 リサイズは後で設定

# 🎯 4️⃣ 姿勢推定モデル
pose_nn = pipeline.createNeuralNetwork()
pose_nn.setBlobPath(pose_model_path)

# 📡 5️⃣ Output nodes
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

xout_face = pipeline.createXLinkOut()
xout_face.setStreamName("face_detection")

xout_pose = pipeline.createXLinkOut()
xout_pose.setStreamName("pose_estimation")

# 🔥 6️⃣ pose_manip に XLinkIn を追加
xin_pose_cfg = pipeline.createXLinkIn()
xin_pose_cfg.setStreamName("pose_manip_config")
xin_pose_cfg.out.link(pose_manip.inputConfig)

# 📌 **修正点:** 顔のバウンディングボックスを `pose_manip` に渡し、60×60 にリサイズ
cam.preview.link(detection_nn.input)  # カメラ → 顔検出
detection_nn.out.link(xout_face.input)  # 顔検出結果を出力

# 🔥 **顔検出のバウンディングボックスを `pose_manip` に送る**
cam.preview.link(pose_manip.inputImage)

# 🔥 **クロップ & 60×60 にリサイズした顔画像を姿勢推定に送る**
pose_manip.out.link(pose_nn.input)
pose_nn.out.link(xout_pose.input)  # 姿勢推定結果を出力

# 🎮 7️⃣ 実行
with dai.Device(pipeline) as device:
    video_q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    face_q = device.getOutputQueue("face_detection", maxSize=4, blocking=False)
    pose_q = device.getOutputQueue("pose_estimation", maxSize=4, blocking=False)
    pose_config_q = device.getInputQueue("pose_manip_config")

    while True:
        in_video = video_q.get()
        frame = in_video.getCvFrame()

        # 👀 顔検出の結果を取得（バウンディングボックス描画）
        in_face = face_q.tryGet()
        if in_face is not None:
            detections = in_face.getFirstLayerFp16()
            if len(detections) > 0:
                for i in range(0, len(detections), 7):
                    conf = detections[i+2]  # 信頼度
                    if conf > 0.5:
                        x1 = detections[i+3]
                        y1 = detections[i+4]
                        x2 = detections[i+5]
                        y2 = detections[i+6]

                        # 🔥 ここでクロップ座標を `pose_manip` に送信
                        cfg = dai.ImageManipConfig()
                        cfg.setCropRect(x1, y1, x2, y2)  
                        cfg.setResize(60, 60)  # 🔥 ここで 60x60 にリサイズ
                        pose_config_q.send(cfg)

                        cv2.rectangle(frame, 
                                      (int(x1 * frame.shape[1]), int(y1 * frame.shape[0])), 
                                      (int(x2 * frame.shape[1]), int(y2 * frame.shape[0])), 
                                      (0, 255, 0), 2)  # 🔥 顔を描画

        # 👀 姿勢推定の結果
        in_pose = pose_q.tryGet()
        if in_pose is not None:
            yaw = in_pose.getLayerFp16("angle_y_fc")[0]  # 🔥 修正
            pitch = in_pose.getLayerFp16("angle_p_fc")[0]  # 🔥 修正
            roll = in_pose.getLayerFp16("angle_r_fc")[0]  # 🔥 修正

            vals = np.array([abs(pitch), abs(yaw), abs(roll)])
            max_index = np.argmax(vals)

            if vals[max_index] < MIN_THRESHOLD:
                txt = "Face Forward"
            elif max_index == 0:
                txt = "Look down" if pitch > 0 else "Look up"
            elif max_index == 1:
                txt = "Turn right" if yaw > 0 else "Turn left"
            elif max_index == 2:
                txt = "Tilt right" if roll > 0 else "Tilt left"

            cv2.putText(frame, f"{txt} (Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f})",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 🎥 映像を表示
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
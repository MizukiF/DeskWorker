import cv2
import depthai as dai
import time
from collections import deque
from MultiMsgSync import TwoStageHostSeqSync
from tools import decode_pose, frame_norm

# グローバル変数
is_deskworking = False
start_time = None
break_time = None
distance_threshold = 1.0  # デスクワークと判定する距離（メートル）
distance_history = deque(maxlen=10)  # 距離の履歴を保持

# パイプラインの作成
from api import create_pipeline
pipeline = create_pipeline(stereo=True)

# デバイスとパイプラインの開始
with dai.Device(pipeline) as device:
    sync = TwoStageHostSeqSync()
    queues = {
        "color": device.getOutputQueue("color", maxSize=4, blocking=False),
        "detection": device.getOutputQueue("detection", maxSize=4, blocking=False),
        "recognition": device.getOutputQueue("recognition", maxSize=4, blocking=False),
    }

    while True:
        # 各キューからメッセージを取得して同期クラスに追加
        for name, q in queues.items():
            if q.has():
                sync.add_msg(q.get(), name)

        # 同期されたメッセージを取得
        msgs = sync.get_msgs()
        if msgs is None:
            continue

        frame = msgs["color"].getCvFrame()
        detections = msgs["detection"].detections
        recognitions = msgs["recognition"]

        detected = False
        distances = []

        # 検出結果を処理
        for i, detection in enumerate(detections):
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            distances.append(detection.spatialCoordinates.z / 1000)  # 距離をメートルに変換
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # 姿勢推定をデコード
            rec = recognitions[i]
            yaw = rec.getLayerFp16('angle_y_fc')[0]
            pitch = rec.getLayerFp16('angle_p_fc')[0]
            roll = rec.getLayerFp16('angle_r_fc')[0]
            decode_pose(yaw, pitch, roll, bbox, frame)

            detected = True

        # 平均距離を計算して表示
        if distances:
            avg_distance = sum(distances) / len(distances)
            distance_history.append(avg_distance)
            smoothed_distance = sum(distance_history) / len(distance_history)
            cv2.putText(frame, f"Smoothed Distance: {smoothed_distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # デスクワーク判定
        if detected and avg_distance < distance_threshold:
            if not is_deskworking:
                start_time = time.time()
                is_deskworking = True
                print(f"Deskwork started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        else:
            if is_deskworking:
                break_time = time.time()
                is_deskworking = False
                print(f"Deskwork breaked at {time.strftime('%H:%M:%S', time.localtime(break_time))}")

        # 画像を表示
        cv2.imshow("OAK-D Lite Face Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
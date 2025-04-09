import cv2
import depthai as dai
import time
from collections import deque
from MultiMsgSync import TwoStageHostSeqSync
from tools import frame_norm

# グローバル変数
is_deskworking = False
start_time = None
break_time = None
bbox_threshold = 0.15  # デスクワークと判定するBBoxの最小サイズ（比率）
detection_start_time = None  # 検出開始時刻
detection_end_time = None  # 検出終了時刻
detection_duration_threshold = 10  # 判定に必要な継続時間（秒）
last_display_time = 0  # 最後にPitch, Yaw, Rollを更新した時刻
last_status_change_time = 0  # 最後に時刻を表示した時刻
status_display_duration = 3  # 時刻を表示する秒数

# パイプラインの作成
from api import create_pipeline
pipeline = create_pipeline()

# デバイスとパイプラインの開始
with dai.Device(pipeline) as device:
    sync = TwoStageHostSeqSync()
    queues = {
        "color": device.getOutputQueue("color", maxSize=4, blocking=False),
        "detection": device.getOutputQueue("detection", maxSize=4, blocking=False),
        "recognition": device.getOutputQueue("recognition", maxSize=4, blocking=False),
    }

    pitch, yaw, roll = None, None, None  # 初期値を設定

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

        # 検出結果を処理
        for i, detection in enumerate(detections):
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            bbox_width = detection.xmax - detection.xmin
            bbox_height = detection.ymax - detection.ymin

            # BBoxのサイズでデスクワーク中かどうかを判定
            if bbox_width > bbox_threshold and bbox_height > bbox_threshold:
                detected = True
                # BBOXを表示
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                # 姿勢推定を3秒おきに更新
                current_time = time.time()
                if current_time - last_display_time >= 3:  # 3秒ごとに更新
                    last_display_time = current_time
                    rec = recognitions[i]
                    yaw = rec.getLayerFp16('angle_y_fc')[0]
                    pitch = rec.getLayerFp16('angle_p_fc')[0]
                    roll = rec.getLayerFp16('angle_r_fc')[0]

        # デスクワーク判定
        current_time = time.time()
        current_time_str = time.strftime('%H:%M:%S', time.localtime(current_time))
        if detected:
            if detection_start_time is None:
                detection_start_time = current_time  # 検出開始時刻を記録
            detection_end_time = None  # 検出終了時刻をリセット

            # 検出が10秒間継続した場合
            if current_time - detection_start_time >= detection_duration_threshold:
                if not is_deskworking:
                    start_time = current_time
                    is_deskworking = True
                    last_status_change_time = current_time
                    print(f"Deskwork started at {current_time_str}")
        else:
            if detection_end_time is None:
                detection_end_time = current_time  # 検出終了時刻を記録
            detection_start_time = None  # 検出開始時刻をリセット

            # 検出が10秒間途切れた場合
            if current_time - detection_end_time >= detection_duration_threshold:
                if is_deskworking:
                    break_time = current_time
                    is_deskworking = False
                    last_status_change_time = current_time
                    print(f"Deskwork breaked at {current_time_str}")

        # 円を表示
        if is_deskworking:
            # 作業中なら青い円
            cv2.circle(frame, (50, 50), 20, (255, 0, 0), -1)
        else:
            # 休憩中なら赤い円
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)

        # 時刻を表示（青または赤に切り替わった直後のみ）
        if current_time - last_status_change_time <= status_display_duration:
            color = (255, 0, 0) if is_deskworking else (0, 0, 255)
            cv2.putText(frame, f"{current_time_str}", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Pitch, Yaw, Rollを表示（常に表示）
        if pitch is not None and yaw is not None and roll is not None:
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # 画像を表示
        cv2.imshow("OAK-D Lite Face Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
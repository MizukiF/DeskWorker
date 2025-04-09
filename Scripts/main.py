import cv2
import depthai as dai
import time
import os
from datetime import datetime
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
last_record_time = time.time()  # 最後に状態を記録した時刻

# ファイル名をフルパスで指定
current_date = datetime.now().strftime("%Y-%m-%d")
file_name = os.path.join(os.getcwd(), f"{current_date}_data.csv")

# デバッグ用出力
print(f"File will be saved as: {file_name}")

# 空のファイルを作成（既に存在する場合は何もしない）
if not os.path.exists(file_name):
    print(f"Creating file: {file_name}")
    with open(file_name, "w") as f:
        f.write("Status,Time\n")  # ヘッダーを追加

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

        detected = False

        # 検出結果を処理
        for detection in detections:
            bbox_width = detection.xmax - detection.xmin
            bbox_height = detection.ymax - detection.ymin

            # BBoxのサイズでデスクワーク中かどうかを判定
            if bbox_width > bbox_threshold and bbox_height > bbox_threshold:
                detected = True

        # デスクワーク判定
        current_time = time.time()
        if detected:
            if detection_start_time is None:
                detection_start_time = current_time  # 検出開始時刻を記録
            detection_end_time = None  # 検出終了時刻をリセット

            # 検出が10秒間継続した場合
            if current_time - detection_start_time >= detection_duration_threshold:
                if not is_deskworking:
                    start_time = current_time
                    is_deskworking = True
                    with open(file_name, "a") as f:
                        f.write(f"Working,{time.strftime('%H:%M:%S', time.localtime(start_time))}\n")
                    print(f"Deskwork started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        else:
            if detection_end_time is None:
                detection_end_time = current_time  # 検出終了時刻を記録
            detection_start_time = None  # 検出開始時刻をリセット

            # 検出が10秒間途切れた場合
            if current_time - detection_end_time >= detection_duration_threshold:
                if is_deskworking:
                    break_time = current_time
                    is_deskworking = False
                    with open(file_name, "a") as f:
                        f.write(f"Break,{time.strftime('%H:%M:%S', time.localtime(break_time))}\n")
                    print(f"Deskwork ended at {time.strftime('%H:%M:%S', time.localtime(break_time))}")

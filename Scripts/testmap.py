import cv2
import depthai as dai

# パイプラインの作成
pipeline = dai.Pipeline()

# ステレオカメラの設定
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)  # 解像度を480Pに設定
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)  # 解像度を480Pに設定
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)  # デフォルトのプロファイルを使用
stereo.setLeftRightCheck(True)  # 左右の整合性チェックを有効化
stereo.setSubpixel(True)        # サブピクセル精度を有効化
stereo.setExtendedDisparity(False)  # 拡張視差を無効化（近距離の精度向上）
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)  # ノイズ除去フィルタを強化
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Depth出力
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# 整列された画像（Rectified Left/Right）の出力
xout_rectified_left = pipeline.create(dai.node.XLinkOut)
xout_rectified_left.setStreamName("rectified_left")
stereo.rectifiedLeft.link(xout_rectified_left.input)

xout_rectified_right = pipeline.create(dai.node.XLinkOut)
xout_rectified_right.setStreamName("rectified_right")
stereo.rectifiedRight.link(xout_rectified_right.input)

# デバイスとパイプラインの開始
with dai.Device(pipeline) as device:
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_rectified_left = device.getOutputQueue(name="rectified_left", maxSize=4, blocking=False)
    q_rectified_right = device.getOutputQueue(name="rectified_right", maxSize=4, blocking=False)

    while True:
        # Depthマップの取得
        depth_frame = q_depth.tryGet()
        if depth_frame is not None:
            depth = depth_frame.getFrame()
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imshow("Depth Map", depth)

        # 整列された左画像の取得
        rectified_left_frame = q_rectified_left.tryGet()
        if rectified_left_frame is not None:
            rectified_left = rectified_left_frame.getCvFrame()
            cv2.imshow("Rectified Left", rectified_left)

        # 整列された右画像の取得
        rectified_right_frame = q_rectified_right.tryGet()
        if rectified_right_frame is not None:
            rectified_right = rectified_right_frame.getCvFrame()
            cv2.imshow("Rectified Right", rectified_right)

        # 終了条件
        if cv2.waitKey(1) == ord('q'):
            break
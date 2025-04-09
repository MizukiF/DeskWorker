from MultiMsgSync import TwoStageHostSeqSync
import cv2
import depthai as dai
import blobconverter
import numpy as np
from pose import frame_norm
from tools import *

def create_pipeline():
    pipeline = dai.Pipeline()
    
    print("Creating color camera...")
    # Color camera node (低解像度に設定)
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)  # 解像度を640x480に設定
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    
    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    cam.preview.link(cam_xout.input)
    
    # ImageManip will resize the frame before sending it to the face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    cam.preview.link(face_det_manip.inputImage)
    
    # Face detection neural network
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.4)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    face_det_manip.out.link(face_det_nn.input)
    
    # Send the detected faces to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    face_det_nn.out.link(face_det_xout.input)
    
    # Script node for synchronization and cropping
    image_manip_script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(image_manip_script.inputs["face_det_in"])
    face_det_nn.passthrough.link(image_manip_script.inputs["passthrough"])
    cam.preview.link(image_manip_script.inputs["preview"])
    
    image_manip_script.setScript("""
    import time
    msgs = dict()

    def add_msg(msg, name, seq=None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg

        if 15 < len(msgs):
            msgs.popitem()  # Remove first element

    def get_msgs():
        global msgs
        seq_remove = []
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq)
            if len(syncMsgs) == 2:  # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                return syncMsgs
        return None

    def correct_bb(xmin, ymin, xmax, ymax):
        if xmin < 0: xmin = 0.001
        if ymin < 0: ymin = 0.001
        if xmax > 1: xmax = 0.999
        if ymax > 1: ymax = 0.999
        return [xmin, ymin, xmax, ymax]

    while True:
        time.sleep(0.001)

        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')

        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
            passthrough = node.io['passthrough'].get()
            seq = passthrough.getSequenceNum()
            add_msg(face_dets, 'dets', seq)

        sync_msgs = get_msgs()
        if sync_msgs is not None:
            img = sync_msgs['preview']
            dets = sync_msgs['dets']
            for det in dets.detections:
                cfg = ImageManipConfig()
                bb = correct_bb(det.xmin - 0.03, det.ymin - 0.03, det.xmax + 0.03, det.ymax + 0.03)
                cfg.setCropRect(*bb)
                cfg.setResize(60, 60)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(img)
    """)

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(60, 60)
    recognition_manip.setWaitForConfigInput(True)
    image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)
    
    # Second stage recognition NN
    print("Creating recognition Neural Network...")
    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
    recognition_manip.out.link(recognition_nn.input)

    recognition_xout = pipeline.create(dai.node.XLinkOut)
    recognition_xout.setStreamName("recognition")
    recognition_nn.out.link(recognition_xout.input)
    
    return pipeline
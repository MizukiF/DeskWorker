from MultiMsgSync import TwoStageHostSeqSync
import cv2
import depthai as dai
import blobconverter
import numpy as np
from pose import frame_norm
from tools import *

def create_pipeline(stereo):
    pipeline = dai.Pipeline()
    
    print("Creating color camera...")
    # Color camera node (produces high-resolution frames)
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1080, 1080)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    
    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    cam.preview.link(cam_xout.input)
    
    copy_manip = pipeline.create(dai.node.ImageManip)
    copy_manip.setNumFramesPool(15)
    copy_manip.setMaxOutputFrameSize(3499200)
    cam.preview.link(copy_manip.inputImage)
    
    # ImageManip will resize the frame before sending it to the face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    copy_manip.out.link(face_det_manip.inputImage)
    
    if stereo:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        
        stereo.setLeftRightCheck(True)  # 左右の整合性チェックを有効化
        stereo.setSubpixel(True)       # サブピクセル精度を有効化
        stereo.setExtendedDisparity(False)  # 拡張視差を無効化（近距離の精度向上）
        
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        
        # Spatial Detection network
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        face_det_nn.setBoundingBoxScaleFactor(0.8)
        face_det_nn.setDepthLowerThreshold(100)
        face_det_nn.setDepthUpperThreshold(5000)
        stereo.depth.link(face_det_nn.inputDepth)
    
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    face_det_manip.out.link(face_det_nn.input)
    
    # Send the detected faces to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    face_det_nn.out.link(face_det_xout.input)
    
    # this script node will take the output from the face detection NN as an input
    # and set ImageManipConfig to the "recognition manip" to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(image_manip_script.inputs["face_det_in"])

    # Only send metadata, we are only interested in timestamp, so we can sync
    # depth frames with NN output
    face_det_nn.passthrough.link(image_manip_script.inputs["passthrough"])
    copy_manip.out.link(image_manip_script.inputs["preview"])
    
    image_manip_script.setScript("""
    import time
    msgs = dict()

    def add_msg(msg, name, seq = None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        # node.warn(f"New msg {name}, seq {seq}")

        # Each seq number has it's own dict of msgs
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg

        # To avoid freezing (not necessary for this ObjDet model)
        if 15 < len(msgs):
            node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() # Remove first element

    def get_msgs():
        global msgs
        seq_remove = [] # Arr of sequence numbers to get deleted
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
            # node.warn(f"Checking sync {seq}")

            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == 2: # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                return syncMsgs # Returned synced msgs
        return None

    def correct_bb(xmin,ymin,xmax,ymax):
        if xmin < 0: xmin = 0.001
        if ymin < 0: ymin = 0.001
        if xmax > 1: xmax = 0.999
        if ymax > 1: ymax = 0.999
        return [xmin,ymin,xmax,ymax]

    while True:
        time.sleep(0.001) # Avoid lazy looping

        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')

        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
            # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
            passthrough = node.io['passthrough'].get()
            seq = passthrough.getSequenceNum()
            add_msg(face_dets, 'dets', seq)

        sync_msgs = get_msgs()
        if sync_msgs is not None:
            img = sync_msgs['preview']
            dets = sync_msgs['dets']
            for i, det in enumerate(dets.detections):
                cfg = ImageManipConfig()
                bb = correct_bb(det.xmin-0.03, det.ymin-0.03, det.xmax+0.03, det.ymax+0.03)
                cfg.setCropRect(*bb)
                # node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
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
    
    # Second stange recognition NN
    print("Creating recognition Neural Network...")
    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
    recognition_manip.out.link(recognition_nn.input)

    recognition_xout = pipeline.create(dai.node.XLinkOut)
    recognition_xout.setStreamName("recognition")
    recognition_nn.out.link(recognition_xout.input)    
    
    return pipeline
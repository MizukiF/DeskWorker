from MultiMsgSync import TwoStageHostSeqSync
import cv2
import depthai as dai
import blobconverter
import numpy as np
from pose import frame_norm
from api import create_pipeline
from tools import *

with dai.Device() as device:
    stereo = 1 < len(device.getConnectedCameras())
    device.startPipeline(create_pipeline(stereo))
    
    
    
    sync = TwoStageHostSeqSync()
    queues = {}
    # Create output queues
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)
    
    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            recognitions = msgs["recognition"]
            
        if detections is not None:
        
        
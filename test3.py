from depthai_sdk import OakCamera, TextPosition, Visualizer
from depthai_sdk.classes.packets  import TwoStagePacket
import numpy as np
import cv2
MIN_THRESHOLD = 15. # Degrees in yaw/pitch/roll to be considered as head movement

with OakCamera() as oak:
    oak.show_graph()
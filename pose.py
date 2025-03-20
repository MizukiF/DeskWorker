import numpy as np
import cv2

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0]) #len(bbow) -> 4 (x_min, y_min, x_max, y_max)
    normVals[::2] = frame.shape[1] # [::2] first to end step by 2. put shapeWidth in 0, 2, 4, ... 2n
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int) # position 0 to 1 -> place images


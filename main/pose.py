import numpy as np
import cv2

def frame_norm(frame, bbox):
    # bboxの長さと同じサイズの配列を作成し、すべての要素をframeの高さで初期化
    normVals = np.full(len(bbox), frame.shape[0]) #len(bbox) must be 4 (x_min, y_min, x_max, y_max)
    
    # 配列の偶数インデックス（x座標）をframeの幅で設定
    normVals[::2] = frame.shape[1] # [::2] first to end step by 2. put shapeWidth in 0, 2, 4, ... 2n
    
    # bboxを0から1の範囲にクリップし、normValsを掛けてスケーリングし、整数に変換
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int) # position 0 to 1 -> place images


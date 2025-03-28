# Depth AI Guide  

## [documentation](https://docs.luxonis.com/software/)
* [Spatial location calc](https://docs.luxonis.com/software/depthai/examples/spatial_location_calculator/)

# Example

## First, import all necessary modules
```
from pathlib import Path

import blobconverter
import cv2
import depthai
import numpy as np
```
## Create a pipeline
```
pipeline = depthai.Pipeline()
```

## Set the Color camera as the output (automatically select the center camera)
```
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)
```
## [Color camera reference](https://docs.luxonis.com/software/depthai-components/nodes/color_camera/)

## Select model
```
detection_nn = pipeline.createMobileNetDetectionNetwork()
```

## Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model<br><br>We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo

```
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
```
## [Compiling a neural network reference](https://docs.luxonis.com/software/ai-inference/conversion/)
## [blobconverter tool](https://github.com/luxonis/blobconverter/tree/master/cli)

## Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0-1>
```
detection_nn.setConfidenceThreshold(0.5)
```

## XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
```
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

cam_rgb.preview.link(xout_rgb.input)
cam_rgb.preview.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)
```

## Pipeline is now finished, and we need to find an available device to run our pipeline<br><br> We are using context manager here that will dispose the device after we stop using it
```
with depthai.Device(pipeline) as device:
```
## From this point, the Device will be in "running" mode and will start sending data via XLink

## To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
## Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    detections = []

## Since the detections returned by nn have values from <0-1> range, they need to be multiplied by frame width/height to receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

## we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
## If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

## when data from nn is received, we take the detections array that contains mobilenet-ssd results
        if in_nn is not None:
            detections = in_nn.detections

## for each bounding box, we first normalize it to match the frame size
        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
## and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
## After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)

## at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break

---
cam_rgb.setInterleaved(True)

に設定すると、カメラのプレビュー出力がインタリーブ形式（YUVフォーマットなど）で出力されます。これにより、後続の処理が複雑になる可能性があります。具体的には、以下のような問題が発生する可能性があります：

データ処理の複雑化: インタリーブ形式のデータを処理するためには、デインターレース処理が必要になることがあります。これにより、処理の複雑さが増し、パフォーマンスが低下する可能性があります。

互換性の問題: 後続のネットワークや処理パイプラインがインタリーブ形式のデータをサポートしていない場合、エラーが発生する可能性があります。

デバッグの困難さ: インタリーブ形式のデータは、デバッグや可視化が難しくなることがあります。特に、画像を直接表示したり保存したりする場合に問題が発生することがあります。

したがって、インタリーブ形式を使用する必要がない場合は、setInterleaved(False) のままにしておく方が無難です。





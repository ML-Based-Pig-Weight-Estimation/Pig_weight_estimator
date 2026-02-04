from ultralytics import YOLO

# Load your trained YOLO detection model
yolo_detect = YOLO("yolo11x.pt")
yolo_detect.export(format="onnx")  # creates yolo11x.onnx

# Load your trained YOLO segmentation model
yolo_seg = YOLO("yolo11x-seg.pt")
yolo_seg.export(format="onnx")  
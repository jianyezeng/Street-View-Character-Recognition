from ultralytics import YOLO
model = YOLO("C:/Users/zjy/Desktop/ultralytics/runs/detect/train21/weights/best.pt")
new_result = model.predict(source="C:/Users/zjy/Desktop/project_implementation/Street-View-Character-Recognition/YOLO/data/test/images")
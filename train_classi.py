from ultralytics import YOLO

model=YOLO("yolov8n.pt")

results =model.train(data="D:\Model\yolov8-silva-main\datasets\Weapons\dmta.yaml",epochs=1)


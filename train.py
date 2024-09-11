try:
    from ultralytics import YOLO
except:
    !pip install ultralytics
import os

# путь до yaml файла, в которым прописаны пути к папкам датасета и информация о метках класса
data_path = r'data.yaml' 

# загрузка модели
model = YOLO(r'yolov8n.pt')

epochs = 400
batch = 64
imgsz = 640

results = model.train(data=data_path,
                  epochs=epochs,
                  batch=batch,
                  imgsz=imgsz,
                  name='red',
                  device='cpu')
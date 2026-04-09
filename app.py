from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
import cv2
import torch
import pandas as pd

app = FastAPI()

# Папка для сохранения загруженных и обработанных изображений
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')
class_names = model.names

@app.post('/detect')
async def detect_objects(file: UploadFile = File(...)):
    # Чтение файла
    contents = await file.read()
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Сохранение файла
    with open(filepath, 'wb') as f:
        f.write(contents)

    # Загрузка изображения для модели
    img = cv2.imread(filepath)
    results = model(img)
    df = results.pandas().xyxy[0]

    # Рисование рамки на изображении
    for _, row in df.iterrows():
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        conf = row['confidence']
        cls = int(row['class'])
        color = (0, 255, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    result_filename = f"{uuid.uuid4().hex}_result.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, img)

    # Возвращение URL обработанного изображения
    return JSONResponse(content={
        'result_image_url': f"/static/uploads/{result_filename}"
    })

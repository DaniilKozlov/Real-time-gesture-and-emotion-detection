from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r'best.pt')

color_map = {
    0: (255, 0, 0),    
    1: (255, 0, 255),    
    2: (0, 255, 0), 
    3: (255, 255, 255),
    4: (255, 51, 102),
    5: (0, 255, 128),
    6: (0, 0, 255)}
capture = cv2.VideoCapture(0)
threshold = 0.5

# для записи видео 
# codec = cv2.VideoWriter_fourcc(*'MPEG')
# output = cv2.VideoWriter('output.mp4', codec, 20.0, (640, 480))

while True:
    ret, frame = capture.read()
    results = model(frame)[0]

    for class_id, box, conf in zip(results.boxes.cls.cpu().numpy(), 
                                    results.boxes.xyxy.cpu().numpy().astype(np.int32),
                                    results.boxes.conf.cpu().numpy()):
        if conf > threshold:  # Проверяем, что уверенность превышает порог
            class_name = results.names[int(class_id)]
            confidence_score = conf.item()  # Преобразуем тензор в число
            
            # Определяем цвет рамки в зависимости от класса
            color = color_map.get(int(class_id), (255, 255, 255))  # По умолчанию белый цвет
            
            x1, y1, x2, y2 = box
            
            # Рисуем рамку вокруг объекта
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Выводим текст с именем класса и вероятностью
            text = f"{class_name}: {confidence_score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # output.write(frame)

    cv2.imshow('Gestures and emotions with YOLOv8', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Выход при нажатии клавиши 'q'
        break

capture.release()
# output.release()
cv2.destroyAllWindows()
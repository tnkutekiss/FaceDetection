import cv2
import numpy as np

# Load dữ liệu cascade cho việc nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load mô hình YOLOv4
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

# Đọc ảnh đầu vào
image = cv2.imread('hinh2.jpg')

# Lấy kích thước ảnh
height, width = image.shape[:2]

# Xây dựng blob từ ảnh đầu vào
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Đưa blob vào mô hình YOLOv4 để nhận diện
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_layers)

# Các bước xử lý các đầu ra của mô hình YOLOv4 để tìm khuôn mặt
class_ids = []
confidences = []
boxes = []

# Xử lý các đầu ra của mô hình YOLOv4
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Kiểm tra xem lớp nhận diện có phải là khuôn mặt không
        if confidence > 0.5 and class_id == 0:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            box_width = int(detection[2] * width)
            box_height = int(detection[3] * height)

            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)

            # Thêm thông tin khuôn mặt vào danh sách
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, box_width, box_height])

# Áp dụng Non-Maximum Suppression để loại bỏ các khuôn mặt trùng lặp
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Hiển thị kết quả
for i in indices:
    x, y, w, h = boxes[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"Confidence: {confidences[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
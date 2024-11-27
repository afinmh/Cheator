import cv2
import os
import torch
import torch.nn as nn
import ultralytics
from pydantic import BaseModel

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        
        if out.dim() == 3:  
            out = out[:, -1, :] 
        elif out.dim() == 2:  
            out = out  

        out = self.fc(out)
        return out

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10

get_keypoint = GetKeypoint()

def extract_keypoint(keypoint):
    return [
        keypoint[get_keypoint.NOSE][0], keypoint[get_keypoint.NOSE][1],
        keypoint[get_keypoint.LEFT_EYE][0], keypoint[get_keypoint.LEFT_EYE][1],
        keypoint[get_keypoint.RIGHT_EYE][0], keypoint[get_keypoint.RIGHT_EYE][1],
        keypoint[get_keypoint.LEFT_EAR][0], keypoint[get_keypoint.LEFT_EAR][1],
        keypoint[get_keypoint.RIGHT_EAR][0], keypoint[get_keypoint.RIGHT_EAR][1],
        keypoint[get_keypoint.LEFT_SHOULDER][0], keypoint[get_keypoint.LEFT_SHOULDER][1],
        keypoint[get_keypoint.RIGHT_SHOULDER][0], keypoint[get_keypoint.RIGHT_SHOULDER][1],
        keypoint[get_keypoint.LEFT_ELBOW][0], keypoint[get_keypoint.LEFT_ELBOW][1],
        keypoint[get_keypoint.RIGHT_ELBOW][0], keypoint[get_keypoint.RIGHT_ELBOW][1],
        keypoint[get_keypoint.LEFT_WRIST][0], keypoint[get_keypoint.LEFT_WRIST][1],
        keypoint[get_keypoint.RIGHT_WRIST][0], keypoint[get_keypoint.RIGHT_WRIST][1]
    ]

model_yolo = ultralytics.YOLO(model='./yolov8s-pose.pt')

input_size = 11 * 2  
hidden_size = 256
num_classes = 2
model_lstm = LSTMNet(input_size, hidden_size, num_classes)
model_lstm.load_state_dict(torch.load(r'D:\Cheator\Cheator-master\accounts\fraud_classification_lstm.pth', map_location=torch.device('cpu')))
model_lstm.eval()  

def detect(frame):
    frame = cv2.flip(frame, 1)
    
    results = model_yolo.predict(frame, save=False)[0]
    results_keypoint = results.keypoints.xyn.cpu().numpy()

    cheating = False
    keypoints_results = []  
    
    for result_keypoint in results_keypoint:
        if len(result_keypoint) == 17:
            keypoint_list = extract_keypoint(result_keypoint)
            keypoints_results.append(keypoint_list)

    input_data = []
    for keypoint_list in keypoints_results:
        input_data.append(keypoint_list)

    if input_data:
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            predictions = model_lstm(input_tensor)

        predicted_classes = torch.argmax(predictions, dim=1)

    for i, bbox in enumerate(results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, bbox) 
        
        if predicted_classes[i] == 0:
            color = (0, 0, 255)
            cheating = True
            text = "Terdeteksi: Cheating"
        else:
            color = (0, 255, 0)
            text = "Terdeteksi: Not cheating"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, cheating
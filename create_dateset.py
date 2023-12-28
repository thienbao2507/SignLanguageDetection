import os
import  pandas as pd
import cv2
import mediapipe as mp

data_dir = './resources/data'

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

# Tạo 1 đối tượng hand để xử lý ảnh tĩnh
hand = mp_hand.Hands(static_image_mode= True, min_detection_confidence= 0.4)

labels = []
data = []

for dir_ in os.listdir(data_dir):
    subfolder_path = os.path.join(data_dir,dir_)

    for img_path in os.listdir(subfolder_path):
        # auxiliary data
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(subfolder_path,img_path))
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        result = hand.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                # Lấy vị trí tuyệt đối của mỗi điểm ảnh
                for i in range(len(hand_landmark.landmark)):
                    x = hand_landmark.landmark[i].x
                    y = hand_landmark.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                #Tính vị trí tương đối
                for i in range(len(hand_landmark.landmark)):
                    x = hand_landmark.landmark[i].x
                    y = hand_landmark.landmark[i].y

                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

df = pd.DataFrame({'data' : data, 'labels' : labels})
df.to_csv('data.csv',index= False)



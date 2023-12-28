import time

import cv2
import os

data_dir = './resources/data'

# Màu xanh dương
text_color = (51, 102, 255)
labels: list[str] = ['Stop', 'Like', 'Dislike', 'Victory']

dataset_size = 100

# Check data_dir is exists or not ,if not will create a new folder
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# If device has a built-in webcam,parameter will be 0.If not,parameter is 1,2,...
cap = cv2.VideoCapture(0)

for i in range(len(labels)):
    # Create subfolders by label name
    subfolder_name = os.path.join(data_dir, labels[i])
    check_dir(subfolder_name)

    print('Collecting data for {}'.format(labels[i]))

    # Wait for the user to press S to start
    while True:
        # Dòng này chụp 1 khung hình từ webcam, result sẽ lưu kết quả(thành công hoặc không)
        # và frame sẽ lưu dữ liệu hình ảnh
        result, frame = cap.read()
        # Lật camera lại
        frame = cv2.flip(frame, 1)
        # thêm dòng text vào góc trái màn hình
        cv2.putText(frame, 'Press S to Start or Q to Quit', (0, 30)
                    , cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3, cv2.LINE_AA)

        # Show màn hình lên
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break
        elif cv2.waitKey(25) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Chụp 100 bức ảnh
    count = 0
    while count < dataset_size:
        result, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('Frame',frame)
        cv2.waitKey(25)
        frame = cv2.flip(frame, 1)
        if result:
            img_name = os.path.join(subfolder_name, '{}_{}.jpg'.format(labels[i].lower(), count))
            cv2.imwrite(img_name, frame)
            count += 1
cap.release()
cv2.destroyAllWindows()
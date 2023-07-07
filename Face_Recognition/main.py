import cv2
import os
from datetime import datetime
cap =  cv2.VideoCapture(0)
while True:
    for i in range(10):
        _, image = cap.read()
        cv2.imshow('Video', image)
        path = 'C:/Users/PC THAO/PycharmProjects/Face_Recognition/Images'
        cv2.imwrite(os.path.join(path, str(i) + '_' + str(datetime.date(datetime.now())) + '.png'), image)
        img  = cv2.imread(str(i) + '_' + str(datetime.date(datetime.now())) + '.png')
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        (humans, _) = hog.detectMultiScale(img, winStride=(10, 10),
                                           padding=(32, 32), scale=1.1)

        # getting no. of human detected
        print('Human Detected : ', len(humans))

        # loop over all detected humans
        for (x, y, w, h) in humans:
            pad_w, pad_h = int(0.15 * w), int(0.01 * h)
            cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
    if cv2.waitKey(1) == ord('q'):
        break

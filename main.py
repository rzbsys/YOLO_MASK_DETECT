import cv2
import torch
import os
import time
vcap = cv2.VideoCapture(0) 


model = torch.hub.load('yolov5', 'custom', path='rmask.pt', source='local')
cnt = 0

while True :
    ret, frame = vcap.read()

    results = model(frame)
    position = results.pandas().xyxy[0].values

    for i in position:
        if i[4] > 0.3:
            if i[5] == 0:
                color = (255, 0, 0)
            elif i[5] == 1:
                color = (0, 0, 255)
            position_xyxy = i[0:4]
            position_xyxy = list(map(int, position_xyxy))

            cv2.rectangle(frame, (position_xyxy[0], position_xyxy[1]), (position_xyxy[2], position_xyxy[3]), color, 1, cv2.LINE_8)
            cv2.putText(frame, i[6] + ' ' + str(i[4] * 100) + '%', (position_xyxy[0], position_xyxy[1] - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            
            if i[5] == 1 and cnt == 0:
                file_len = len(os.listdir('img'))
                cv2.imwrite('img/%d.png' % (file_len + 1), frame)
                cnt = 10
    if cnt > 0:
        cnt = cnt - 1
        print(cnt)
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(1) == 27 :
        vcap.release() 
        cv2.destroyAllWindows()
        break
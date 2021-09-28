import cv2
import numpy as np
from edge_detection import edge_detection
from canny import *
from knn import *
from MOG2_contours import *

video_path = "./video/Badminton.mp4"

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = int(round(cap.get(cv2.CAP_PROP_FPS)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))        # 取得畫面尺寸
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ratio = width / height

# WIDTH = 1280
# HEIGHT = int(WIDTH / ratio)

# 建立 VideoWriter 物件，輸出影片至 output.mp4，FPS 值為 30.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')             # 使用 XVID 編碼
out = cv2.VideoWriter('./video/HomeWork_Badminton.mp4', fourcc, 30.0, (width, height), True)

def add_text(frame, used_method):
    for i, test in enumerate(used_method):
        cv2.putText(frame, test, (50, 450 + 40*i), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (875, 90), (1280, 0), (255, 255, 255), -1)
    cv2.putText(frame, '  Tokyo 2020 Olympic', (880, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Badminton Women single', (880, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

def add_dialog(frame, square, who, content):
    # 0 : Intanon
    # 1 : Tai Tzu-ying
    content = content.split(' ')
    if int(who) == 1:
        (x,y,w,h) = square[1]
        triangle_cnt = np.array( [(x+w, y+40), (x+w+10, y+10), (x+w+30, y+30)], np.int32 )
        triangle_cnt = triangle_cnt.reshape((-1,1,2))
        cv2.fillPoly(frame, [triangle_cnt], (255,255,255))
        cv2.ellipse(frame, (x+w+70, y), (70, 50), 0, 0, 360, (255, 255, 255), -1)
        for i,word in enumerate(content):
            cv2.putText(frame, word, (x+w+20, y+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    elif int(who) == 0:
        (x,y,w,h) = square[0]
        triangle_cnt = np.array( [(x+w, y+40), (x+w+10, y+10), (x+w+30, y+30)], np.int32 )
        triangle_cnt = triangle_cnt.reshape((-1,1,2))
        cv2.fillPoly(frame, [triangle_cnt], (255,255,255))
        cv2.ellipse(frame, (x+w+70, y), (70, 50), 0, 0, 360, (255, 255, 255), -1)
        for i,word in enumerate(content):
            cv2.putText(frame, word, (x+w+20, y+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    else:
        print('Wrong code for player!') 


used_method = ['1.put text','2.flip','3.gray','4.KNN','5.Canny','6.Contours','7.sobel']
cuts = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # gray
    if cuts > FPS*1 and cuts <= FPS*3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = MOG2_contours(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        add_text(frame, used_method)

        out.write(frame)                               # 寫入影格
        cv2.imshow('frame', frame)

    # KNN
    elif cuts > FPS*3 and cuts <= FPS*6:
        frame = knn(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        add_text(frame, used_method)

        out.write(frame)                               # 寫入影格
        cv2.imshow('frame', frame)

    # Canny
    elif cuts > FPS*6 and cuts <= FPS*9:
        frame_tmp = frame
        square = dialog(frame_tmp)
        frame = canny(frame, 100, 200)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if (cuts >= 240 and cuts <= 270) and square != None:
            add_dialog(frame, square, 1, 'Fake Action')
        if (cuts >= 260 and cuts <= 290) and square != None:
            add_dialog(frame, square, 0, 'Chance Ball')

        add_text(frame, used_method)

        out.write(frame)                               # 寫入影格
        cv2.imshow('frame', frame)

    # MOG2_contours  
    elif cuts > FPS*9 and cuts <= FPS*12:
        frame_tmp = frame
        square = dialog(frame_tmp)
        frame = MOG2_contours(frame)

        if (cuts >= 270 and cuts <= 280) and square != None:
            add_dialog(frame, square, 1, 'Fake Action')
        if (cuts >= 270 and cuts <= 300) and square != None:
            add_dialog(frame, square, 0, 'Chance Ball')
        if (cuts >= 335 and cuts <= 375) and square != None:
            add_dialog(frame, square, 0, 'Smash!')

        add_text(frame, used_method)

        out.write(frame)                               # 寫入影格
        cv2.imshow('frame', frame)

    # edge_detection sobel
    elif cuts > FPS*12 and cuts <= frame_count:
        frame_tmp = frame
        square = dialog(frame_tmp)
        frame = edge_detection(frame)

        if (cuts >= 360 and cuts <= 400) and square != None:
            add_dialog(frame, square, 1, 'Magic Shot')
        if (cuts >= 390 and cuts <= 430) and square != None:
            add_dialog(frame, square, 0, 'Oh~ No~')

        add_text(frame, used_method)

        out.write(frame)                               # 寫入影格
        cv2.imshow('frame', frame)

    # original
    else:
        add_text(frame, used_method)
        out.write(frame)                               # 寫入影格
        cv2.imshow('frame', frame)

    # if (cuts >= 240 and cuts <= 275) or (cuts >= 350 and cuts <= 360):
    #     k = cv2.waitKey(100) & 0xff
    k = cv2.waitKey(1) & 0xff    # 若有按按鍵將回傳該按鍵的 ASCII
    if k == 27:   # Esc ASCII = 27
        break        
    cuts += 1

cap.release()
out.release()
cv2.destroyAllWindows()

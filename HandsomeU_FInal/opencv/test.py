import requests
# If you are using a Jupyter notebook, uncomment the following line.
#%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import numpy as np
import cv2
import time
import random

def face_recognition(frame):
    cv2.imwrite('saveimage/emotion.jpg', frame)
    path = 'saveimage/emotion.jpg'

    subscription_key = "aea68bdca14d4bb38460341b4db77c0f"
    face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'

    image_path = path
    image_data = open(image_path, "rb").read()

    headers = {'Ocp-Apim-Subscription-Key': subscription_key , "Content-Type": "application/octet-stream"}
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
        'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
    }

    data = {'url': image_path}

    response = requests.post(face_api_url, params=params, headers=headers, data = image_data)
    face = response.json()

    if face == []:
        return "NOTFOUNDFACE"

# Display the original image and overlay it with the face information.
    image = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    ax = plt.imshow(image, alpha=0.6)
    #for face in faces:
    fr = face[0]["faceRectangle"]
    fa = face[0]["faceAttributes"]
    emotion = fa["emotion"]

    anger = emotion["anger"]
    contempt = emotion["contempt"]
    disgust = emotion["disgust"]
    fear = emotion["fear"]
    happiness = emotion["happiness"]
    neutral = emotion["neutral"]
    sadness = emotion["sadness"]
    surprise = emotion["surprise"]

    origin = (fr["left"], fr["top"])
    p = patches.Rectangle(
    origin, fr["width"], fr["height"], fill=False, linewidth=2, color='b')
    ax.axes.add_patch(p)
    plt.text(origin[0], origin[1], "%s, %d"%(fa["gender"].capitalize(), fa["age"]),
    fontsize=20, weight="bold", va="bottom")
    _ = plt.axis("off")

    #plt.show()

    MaxValue =  max(anger, contempt , disgust , fear , happiness , neutral , sadness , surprise)

    if MaxValue == anger:
        return "ANGER"
    elif MaxValue == contempt:
        return "CONTEMPT"
    elif MaxValue == disgust:
        return "DISGUST"
    elif MaxValue == happiness:
        return "HAPPINESS"
    elif MaxValue == neutral:
        return "NEUTRAL"
    elif MaxValue == sadness:
        return "SADNESS"
    elif MaxValue == surprise:
        return "SURPRISE"

def easymode():
    before_emotion_order_1 = 0
    before_emotion_order_2 = 0
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)
    life = 5
    score = 0
    count = 0
    combo = 0
    each_score = 100
    add_score = 10

    while(life != 0):
        emotion_order = random.randrange(1,4)
        while(1):
            emotion_order = random.randrange(1, 4)
            if(emotion_order != before_emotion_order_1):
                if(emotion_order != before_emotion_order_2):
                    break

        before_emotion_order_2 = before_emotion_order_1
        before_emotion_order_1 = emotion_order

        if(emotion_order == 1):
            emotion_order = "ANGER"
        elif (emotion_order == 2):
            emotion_order = "SURPRISE"
        elif (emotion_order == 3):
            emotion_order = "HAPPINESS"
        elif (emotion_order == 4):
            emotion_order = "SADNESS"
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        start_time = time.time()
        while ((int)(time.time()- start_time) < 3):
            cv2.putText(frame, "Score : " + str(score), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, "Life : " + str(life), (1000, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, "Subject : " + emotion_order, (10, 680), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, str(3 - (int)(time.time()- start_time)), (550, 400), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 255, 255), 2)
            cv2.imshow('Game', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)

        result_recognition = face_recognition(frame)

        if (emotion_order == result_recognition):
            score += each_score
            count += 1
            combo += 1
            each_score += add_score
        elif (result_recognition == "NOTFOUNDFACE"):
            life = life
        else:
            combo = 0
            each_score = 100
            life -= 1

        start_time = time.time()
        while ((int)(time.time() - start_time) < 3):
            if (emotion_order == result_recognition):
                cv2.putText(frame, "True Answer", (300, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            elif (result_recognition == "NOTFOUNDFACE"):
                cv2.putText(frame, "Not Found Face", (200, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "False Answer", (280, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)

            cv2.putText(frame, "Score : " + str(score), (10,50), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2)
            cv2.putText(frame, "Life : " + str(life), (1000, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, "Subject : " + emotion_order, (10, 650), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            if(result_recognition != "NOTFOUNDFACE"):
                cv2.putText(frame, "Face : " + result_recognition, (700, 650), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.imshow('Game', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    return score


def hardmode():
    before_emotion_order_1 = 0
    before_emotion_order_2 = 0
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)

    life = 3
    score = 0
    count = 0
    combo = 0
    each_score = 100
    add_score = 10

    while(life != 0):
        emotion_order = random.randrange(1,7)
        while(1):
            emotion_order = random.randrange(1, 7)
            if(emotion_order != before_emotion_order_1):
                if(emotion_order != before_emotion_order_2):
                    break

        before_emotion_order_2 = before_emotion_order_1
        before_emotion_order_1 = emotion_order

        if(emotion_order == 1):
            emotion_order = "ANGER"
        elif (emotion_order == 2):
            emotion_order = "CONTEMPT"
        elif (emotion_order == 3):
            emotion_order = "DISGUST"
        elif (emotion_order == 4):
            emotion_order = "HAPPINESS"
        elif (emotion_order == 5):
            emotion_order = "NEUTRAL"
        elif (emotion_order == 6):
            emotion_order = "SADNESS"
        elif (emotion_order == 7):
            emotion_order = "SURPRISE"

        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        start_time = time.time()
        while ((int)(time.time()- start_time) < 3):
            cv2.putText(frame, "Score : " + str(score), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, "Life : " + str(life), (1000, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, "Subject : " + emotion_order, (10, 680), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, str(3 - (int)(time.time()- start_time)), (550, 400), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 255, 255), 2)
            cv2.imshow('Game', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)

        result_recognition = face_recognition(frame)

        if (emotion_order == result_recognition):
            score += each_score
            count += 1
            combo += 1
            each_score += add_score
        elif (result_recognition == "NOTFOUNDFACE"):
            life = life
        else:
            combo = 0
            each_score = 100
            life -= 1

        start_time = time.time()
        while ((int)(time.time() - start_time) < 3):
            if (emotion_order == result_recognition):
                cv2.putText(frame, "True Answer", (300, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            elif (result_recognition == "NOTFOUNDFACE"):
                cv2.putText(frame, "Not Found Face", (200, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "False Answer", (280, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)

            cv2.putText(frame, "Score : " + str(score), (10,50), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2)
            cv2.putText(frame, "Life : " + str(life), (1000, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, "Subject : " + emotion_order, (10, 650), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            if (result_recognition != "NOTFOUNDFACE"):
                cv2.putText(frame, "Face : " + result_recognition, (700, 650), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.imshow('Game', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    return score

image = 0
pt_x = 0
click = 0

title = cv2.imread('usedimage/title4.png', cv2.IMREAD_COLOR)
line_pic = cv2.imread('usedimage//Boundary.png', cv2.IMREAD_COLOR)
score = cv2.imread('usedimage//LeaderBoard.png', cv2.IMREAD_COLOR)
mode_ez = cv2.imread('usedimage//ez_mode.png', cv2.IMREAD_COLOR)
mode_hd = cv2.imread('usedimage//hd_mode.png', cv2.IMREAD_COLOR)

def resize_pics() :
    global mode_ez,mode_hd,line_pic,score,title

    mode_ez = cv2.resize(mode_ez, (80, 80))
    mode_hd = cv2.resize(mode_hd, (80, 80))
    line_pic = cv2.resize(line_pic, (1, 210))
    score = cv2.resize(score, (300, 150))
    title = cv2.resize(title, None, fx = 0.5, fy = 0.5)

def pics() :
    global line_pic, title, score, image,mode_ez,mode_hd

    rows1, cols1, garbage = line_pic.shape
    rows2, cols2, garbage2 = title.shape
    rows3, cols3, garbage3 = score.shape

    for i in range(40, rows1) :
        for j in range(cols1) :
            b = line_pic.item(i, j, 0)
            g = line_pic.item(i, j, 1)
            r = line_pic.item(i, j, 2)

            image.itemset((i, j + 320, 0), b)
            image.itemset((i, j + 320, 1), g)
            image.itemset((i, j + 320, 2), r)

    for i in range(rows2) :
        for j in range(cols2) :
            b = title.item(i, j, 0)
            g = title.item(i, j, 1)
            r = title.item(i, j, 2)

            image.itemset((i, j + 165, 0), b)
            image.itemset((i, j + 165, 1), g)
            image.itemset((i, j + 165, 2), r)

    for i in range(rows3) :
        for j in range(cols3) :
            b = score.item(i, j, 0)
            g = score.item(i, j, 1)
            r = score.item(i, j, 2)

            image.itemset((i+210, j + 165, 0), b)
            image.itemset((i+210, j + 165, 1), g)
            image.itemset((i+210, j + 165, 2), r)

def ezmode():
    global mode_ez, image

    roi = image[150:230, 0:80]
    img2gray = cv2.cvtColor(mode_ez, cv2.COLOR_BGR2GRAY)
    if pt_x < 320:
        scal = 150
    else:
        scal = 255
    ret2, mask = cv2.threshold(img2gray, 10, scal, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(mode_ez, mode_ez, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    image[150:230, 0:80] = dst

def hdmode():
    global mode_hd, image

    roi = image[150:230, 560:640]
    img2gray = cv2.cvtColor(mode_hd, cv2.COLOR_BGR2GRAY)
    if pt_x > 320:
        scal = 150
    else:
        scal = 255
    ret2, mask = cv2.threshold(img2gray, 10, scal, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(mode_hd, mode_hd, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    image[150:230, 560:640] = dst

def onMouse(event, x, y, flag, param):
    global click, pt_x

    if event == cv2.EVENT_MOUSEMOVE:
        pt_x = x

    if event == cv2.EVENT_LBUTTONDOWN:
        click = 1

    if event == cv2.EVENT_LBUTTONUP:
        click = 0

resize_pics()

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

best_score = 0
each_score = 0
best_time = time.localtime()

while(1):
    cv2.setMouseCallback("image", onMouse, 0)

    ret, image = capture.read()
    image = cv2.flip(image, 1)

    pics()
    ezmode()
    hdmode()

    if pt_x > 320 and click == 1:
        capture.release()
        cv2.destroyAllWindows()
        each_score = hardmode()
        best_time = time.localtime()
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        click = 0
    if pt_x < 320 and click == 1:
        capture.release()
        cv2.destroyAllWindows()
        each_score = easymode()
        best_time = time.localtime()
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        click = 0

    if(each_score >= best_score):
        best_score = each_score

    #print_time = "%04d-%02d-%02d %02d:%02d:%02d" % (now_time.tm_year, now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min, now_time.tm_sec)
    print_time = "%02d:%02d:%02d" % (best_time.tm_hour, best_time.tm_min, best_time.tm_sec)

    if(best_score != 0):
        cv2.putText(image, "Time", (325, 265), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, print_time, (325, 290), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "Score", (325, 322), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, str(best_score), (325, 355), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
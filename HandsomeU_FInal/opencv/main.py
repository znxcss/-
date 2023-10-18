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

frame = 0
handsomeu = 0
title = cv2.imread('../image/title4.png', cv2.IMREAD_COLOR)
line_pic = cv2.imread('../image/Boundary.png', cv2.IMREAD_COLOR)
score = cv2.imread('../image/LeaderBoard.png', cv2.IMREAD_COLOR)
mode_ez = cv2.imread('../image/ez_mode.png', cv2.IMREAD_COLOR)
mode_hd = cv2.imread('../image/hd_mode.png', cv2.IMREAD_COLOR)
mode_ez2 = cv2.imread('../image/ez_mode.png', cv2.IMREAD_COLOR)
mode_hd2 = cv2.imread('../image/hd_mode.png', cv2.IMREAD_COLOR)
heart = cv2.imread('../image/Life.png', cv2.IMREAD_COLOR)
noheart = cv2.imread('../image/LowLife.png', cv2.IMREAD_COLOR)

anger_pic = cv2.imread('../image/Anger.png', cv2.IMREAD_COLOR)
cont_pic = cv2.imread('../image/Contempt.png', cv2.IMREAD_COLOR)
hap_pic = cv2.imread('../image/Happiness.png', cv2.IMREAD_COLOR)
sad_pic = cv2.imread('../image/Sad.png', cv2.IMREAD_COLOR)
surp_pic = cv2.imread('../image/Surprised.png', cv2.IMREAD_COLOR)
neu_pic = cv2.imread('../image/Neutral.png', cv2.IMREAD_COLOR)
dis_pic = cv2.imread('../image/Disgust.png', cv2.IMREAD_COLOR)
fear_pic = cv2.imread('../image/Fear.png', cv2.IMREAD_COLOR)

zero_pic = cv2.imread('../image/0.png', cv2.IMREAD_COLOR)
one_pic = cv2.imread('../image/1.png', cv2.IMREAD_COLOR)
two_pic = cv2.imread('../image/2.png', cv2.IMREAD_COLOR)
three_pic = cv2.imread('../image/3.png', cv2.IMREAD_COLOR)
four_pic = cv2.imread('../image/4.png', cv2.IMREAD_COLOR)
five_pic = cv2.imread('../image/5.png', cv2.IMREAD_COLOR)
six_pic = cv2.imread('../image/6.png', cv2.IMREAD_COLOR)
seven_pic = cv2.imread('../image/7.png', cv2.IMREAD_COLOR)
eight_pic = cv2.imread('../image/8.png', cv2.IMREAD_COLOR)
nine_pic = cv2.imread('../image/9.png', cv2.IMREAD_COLOR)
black = cv2.imread('../image/black.png', cv2.IMREAD_COLOR)

def resize_pics() :
    global mode_ez,mode_hd,line_pic,score,title,heart,anger_pic,cont_pic,hap_pic,surp_pic,sad_pic, neu_pic, dis_pic, mode_hd2, mode_ez2, zero_pic, one_pic, two_pic, three_pic, four_pic, five_pic, six_pic, seven_pic, eight_pic, nine_pic,noheart, black, fear_pic

    mode_ez = cv2.resize(mode_ez, (80, 80))
    mode_hd = cv2.resize(mode_hd, (80, 80))
    mode_ez2 = cv2.resize(mode_ez2, (150, 150))
    mode_hd2 = cv2.resize(mode_hd2, (150, 150))
    line_pic = cv2.resize(line_pic, (1, 210))
    score = cv2.resize(score, (300, 150))
    title = cv2.resize(title, None, fx = 0.5, fy = 0.5)
    heart = cv2.resize(heart, (50, 50))
    anger_pic = cv2.resize(anger_pic, (320,180))
    cont_pic = cv2.resize(cont_pic, (320, 180))
    sad_pic = cv2.resize(sad_pic, (320, 180))
    hap_pic = cv2.resize(hap_pic, (320, 180))
    surp_pic = cv2.resize(surp_pic, (320, 180))
    neu_pic = cv2.resize(neu_pic, (320, 180))
    dis_pic = cv2.resize(dis_pic, (320, 180))
    fear_pic = cv2.resize(fear_pic, (320, 180))
    zero_pic = cv2.resize(zero_pic, (50,100))
    one_pic = cv2.resize(one_pic, (50,100))
    two_pic = cv2.resize(two_pic, (50, 100))
    three_pic = cv2.resize(three_pic, (50, 100))
    four_pic = cv2.resize(four_pic, (50, 100))
    five_pic = cv2.resize(five_pic, (50, 100))
    six_pic = cv2.resize(six_pic, (50, 100))
    seven_pic = cv2.resize(seven_pic, (50, 100))
    eight_pic = cv2.resize(eight_pic, (50, 100))
    nine_pic = cv2.resize(nine_pic, (50, 100))
    noheart = cv2.resize(noheart, (50, 50))
    black = cv2.resize(black, (50,50))

resize_pics()
emo_pics = [anger_pic, surp_pic, hap_pic, sad_pic, neu_pic, dis_pic, cont_pic, fear_pic]
numbers = [zero_pic, one_pic, two_pic, three_pic, four_pic, five_pic, six_pic, seven_pic, eight_pic, nine_pic]


def face_recognition():
    global frame

    cv2.imwrite('saveimage/emotion.jpg', frame)
    path = 'saveimage/emotion.jpg'

    subscription_key = "d8b2fc1746fe433e9e8f6b981e424d8c"
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
    image2 = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    ax = plt.imshow(image2, alpha=0.6)
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

def life1(isornot):
    global frame, heart, noheart, black

    if isornot == 1:
        roi = frame[650:700, 20:70]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life2(0)

    elif isornot == 2:
        roi = frame[650:700, 20:70]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        #life2(3)
        #life3(3)
    elif isornot == 3:
        roi = frame[650:700, 20:70]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life2(3)
        life3(3)

    else :
        roi = frame[650:700, 20:70]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life2(0)

    frame[650:700, 20:70] = dst

def life2(isornot):
    global frame, heart, noheart, handsomeu
    handsomeu = cv2.imread('saveimage/emotion.jpg', cv2.IMREAD_COLOR)

    if isornot == 1:
        roi = frame[650:700, 75:125]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life1(1)
        life3(0)
    elif isornot == 2:
        roi = frame[650:700, 75:125]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life1(2)
        life3(3)
    elif isornot == 3:
        roi = frame[650:700, 75:125]
        img2gray = cv2.cvtColor(noheart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(noheart, noheart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life1(2)
        life3(3)
    else :
        roi = frame[650:700, 75:125]
        img2gray = cv2.cvtColor(noheart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(noheart, noheart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life3(0)

    frame[650:700, 75:125] = dst

def life3(isornot):
    global frame, heart, noheart

    if isornot == 1:
        roi = frame[650:700, 130:180]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life2(1)
        life4(0)

    elif isornot == 2:
        roi = frame[650:700, 130:180]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life2(2)
        life4(2)

    elif isornot == 3:
        roi = frame[650:700, 130:180]
        img2gray = cv2.cvtColor(noheart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(noheart, noheart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)

    else :
        roi = frame[650:700, 130:180]
        img2gray = cv2.cvtColor(noheart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(noheart, noheart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life4(0)

    frame[650:700, 130:180] = dst

def life4(isornot):
    global frame, heart, black, noheart

    if isornot == 1:
        roi = frame[650:700, 185:235]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life3(1)
        life5(0)

    elif isornot == 2:
        roi = frame[650:700, 185:235]
        img2gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(black, black, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life5(2)
    else :
        roi = frame[650:700, 185:235]
        img2gray = cv2.cvtColor(noheart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(noheart, noheart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life5(0)

    frame[650:700, 185:235] = dst


def life5(isornot):
    global frame, heart, noheart, black

    if isornot == 1:
        roi = frame[650:700, 240:290]
        img2gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(heart, heart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)
        life4(1)

    elif isornot == 2:
        roi = frame[650:700, 240:290]
        img2gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(black, black, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)

    else :
        roi = frame[650:700, 240:290]
        img2gray = cv2.cvtColor(noheart, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_fg = cv2.bitwise_and(noheart, noheart, mask=mask)
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        dst = cv2.add(img1_fg, img2_bg)

    frame[650:700, 240:290] = dst

def ezmode2():
    global mode_ez2, frame

    roi = frame[20:170, 20:170]
    img2gray = cv2.cvtColor(mode_ez2, cv2.COLOR_BGR2GRAY)

    ret2, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(mode_ez2, mode_ez2, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    frame[20:170, 20:170] = dst

def hdmode2():
    global mode_hd2, frame

    roi = frame[20:170, 20:170]
    img2gray = cv2.cvtColor(mode_hd2, cv2.COLOR_BGR2GRAY)

    ret2, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(mode_hd2, mode_hd2, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    frame[20:170, 20:170] = dst

def emotions(add):
    global frame, emo_pics
    flag = add

    roi = frame[540:720, 960:1280]

    img2gray = cv2.cvtColor(emo_pics[flag], cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(emo_pics[flag], emo_pics[flag], mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    frame[540:720, 960:1280] = dst

def scorehund(scoreval) :
    global frame, numbers

    hund = scoreval/100

    roi = frame[20:120, 1060:1110]

    img2gray = cv2.cvtColor(numbers[hund], cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(numbers[hund], numbers[hund], mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    frame[20:120, 1060:1110] = dst

def scoreten(scoreval) :
    global frame, numbers

    ten = (scoreval%100)/10

    roi = frame[20:120, 1110:1160]

    img2gray = cv2.cvtColor(numbers[ten], cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(numbers[ten], numbers[ten], mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    frame[20:120, 1110:1160] = dst

def scoreone(scoreval) :
    global frame

    one = scoreval%10

    roi = frame[20:120, 1160:1210]

    img2gray = cv2.cvtColor(numbers[one], cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(numbers[one], numbers[one], mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(img1_fg, img2_bg)

    frame[20:120, 1160:1210] = dst

def normalmode():
    global frame
    before_emotion_order_1 = 0
    before_emotion_order_2 = 0

    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)

    life = 5
    score = 0
    count = 0
    combo = 0
    each_score = 10
    add_score = 1
    result_recognition = "NONE"

    while(life != 0):
        random.seed(time.time())
        if (result_recognition != "NOTFOUNDFACE"):
            emotion_order = random.randrange(1,6)
            while (1):
                emotion_order = random.randrange(1, 6)
                if (emotion_order != before_emotion_order_1):
                    if (emotion_order != before_emotion_order_2):
                        break

            before_emotion_order_2 = before_emotion_order_1
            before_emotion_order_1 = emotion_order

        if(emotion_order == 1):
            emotion_order = "ANGER"
            emonum = 0
        elif (emotion_order == 2):
            emotion_order = "SURPRISE"
            emonum = 1
        elif (emotion_order == 3):
            emotion_order = "HAPPINESS"
            emonum = 2
        elif (emotion_order == 4):
            emotion_order = "SADNESS"
            emonum = 3
        elif (emotion_order == 5):
            emotion_order = "FEAR"
            emonum = 7
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        start_time = time.time()
        while ((int)(time.time()- start_time) < 3):
            emotions(emonum)
            ezmode2()
            if life == 1 :
                life1(1)
            elif life == 2:
                life2(1)
            elif life == 3:
                life3(1)
            elif life == 4:
                life4(1)
            elif life == 5:
                life5(1)
            scorehund(score)
            scoreten(score)
            scoreone(score)
            cv2.putText(frame, str(3 - (int)(time.time()- start_time)), (550, 400), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 255, 255), 2)
            cv2.imshow('Game', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)

        result_recognition = face_recognition()

        if (emotion_order == result_recognition):
            score += each_score
            count += 1
            combo += 1
            each_score += add_score
        elif (result_recognition == "NOTFOUNDFACE"):
            life = life
        else:
            combo = 0
            each_score = 10
            life -= 1

        start_time = time.time()
        while ((int)(time.time() - start_time) < 5):
            if (emotion_order == result_recognition):
                cv2.putText(frame, "True Answer", (300, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            elif (result_recognition == "NOTFOUNDFACE"):
                cv2.putText(frame, "Not Found Face", (200, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "False Answer", (280, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)

            emotions(emonum)
            ezmode2()
            if life == 1 :
                life1(1)
            elif life == 2:
                life2(1)
            elif life == 3:
                life3(1)
            elif life == 4:
                life4(1)
            elif life == 5:
                life5(1)
            scorehund(score)
            scoreten(score)
            scoreone(score)
            if (emotion_order != result_recognition):
                if(result_recognition != "NOTFOUNDFACE"):
                    cv2.putText(frame,"Face : " +  result_recognition, (350, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.imshow('Game', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    return score


def hardmode():
    global frame

    before_emotion_order_1 = 0
    before_emotion_order_2 = 0
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)

    life = 3
    score = 0
    count = 0
    combo = 0
    each_score = 30
    add_score = 3
    result_recognition = "NONE"

    while(life != 0):
        random.seed(time.time())
        if(result_recognition != "NOTFOUNDFACE"):
            emotion_order = random.randrange(1,9)
            while (1):
                emotion_order = random.randrange(1, 9)
                if (emotion_order != before_emotion_order_1):
                    if (emotion_order != before_emotion_order_2):
                        break

            before_emotion_order_2 = before_emotion_order_1
            before_emotion_order_1 = emotion_order

        if(emotion_order == 1):
            emotion_order = "ANGER"
            emonum = 0
        elif (emotion_order == 2):
            emotion_order = "SURPRISE"
            emonum = 1
        elif (emotion_order == 3):
            emotion_order = "HAPPINESS"
            emonum = 2
        elif (emotion_order == 4):
            emotion_order = "SADNESS"
            emonum = 3
        elif (emotion_order == 5):
            emotion_order = "NEUTRAL"
            emonum = 4
        elif (emotion_order == 6):
            emotion_order = "DISGUST"
            emonum = 5
        elif (emotion_order == 7):
            emotion_order = "CONTEMPT"
            emonum = 6
        elif (emotion_order == 8):
            emotion_order = "FEAR"
            emonum = 7


        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        start_time = time.time()
        while ((int)(time.time()- start_time) < 3):
            emotions(emonum)
            hdmode2()
            if life == 1:
                life1(3)
            elif life == 2:
                life2(2)
            elif life == 3:
                life3(2)

            scorehund(score)
            scoreten(score)
            scoreone(score)
            #cv2.putText(frame, "Score : " + str(score), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            #cv2.putText(frame, "Life : " + str(life), (1000, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            #cv2.putText(frame, "Subject : " + emotion_order, (10, 680), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(frame, str(3 - (int)(time.time()- start_time)), (550, 400), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 255, 255), 2)
            cv2.imshow('Game', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)

        result_recognition = face_recognition()

        if (emotion_order == result_recognition):
            score += each_score
            count += 1
            combo += 1
            each_score += add_score
        elif (result_recognition == "NOTFOUNDFACE"):
            life = life
        else:
            combo = 0
            each_score = 30
            life -= 1

        start_time = time.time()
        while ((int)(time.time() - start_time) < 5):
            if (emotion_order == result_recognition):
                cv2.putText(frame, "True Answer", (300, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            elif (result_recognition == "NOTFOUNDFACE"):
                cv2.putText(frame, "Not Found Face", (200, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "False Answer", (280, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)

            emotions(emonum)
            hdmode2()
            emotions(emonum)
            hdmode2()
            if life == 1:
                life1(3)
            elif life == 2:
                life2(2)
            elif life == 3:
                life3(2)
            scorehund(score)
            scoreten(score)
            scoreone(score)
            if (emotion_order != result_recognition):
                if(result_recognition != "NOTFOUNDFACE"):
                    cv2.putText(frame,"Face : " +  result_recognition, (350, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.imshow('Game', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                breakq

    capture.release()
    cv2.destroyAllWindows()
    return score

image = 0
pt_x = 0
click = 0

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

best_score = 0
each_score = 0
best_time = time.localtime()
top_ranker = 0

while(1):
    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

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
            capture = cv2.VideoCapture(0)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            click = 0
        if pt_x < 320 and click == 1:
            capture.release()
            cv2.destroyAllWindows()
            each_score = normalmode()
            capture = cv2.VideoCapture(0)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            click = 0

        if (each_score >= best_score):
            best_score = each_score
            top_ranker = handsomeu
            top_ranker = cv2.resize(top_ranker, (130,90))

        print_time = "%02d:%02d:%02d" % (best_time.tm_hour, best_time.tm_min, best_time.tm_sec)

        if (best_score != 0):
            roi = image[255:345, 180:310]
            img2gray = cv2.cvtColor(top_ranker, cv2.COLOR_BGR2GRAY)

            ret2, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img1_fg = cv2.bitwise_and(top_ranker, top_ranker, mask=mask)
            img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            dst = cv2.add(img1_fg, img2_bg)

            image[255:345, 180:310] = dst
            cv2.putText(image, "Time", (325, 265), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, print_time, (325, 290), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, "Score", (325, 322), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, str(best_score), (325, 355), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow('image', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
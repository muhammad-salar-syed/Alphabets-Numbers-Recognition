import cv2
import mediapipe as mp
import time
import pickle
import numpy as np
import streamlit as st

def detectHand(img):
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        myHand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            # print(id, cx, cy)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    return img, lmList

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(min_detection_confidence=0.7)
tipIds = [4, 8, 12, 16, 20]

_, col2, _ = st.columns([1, 12, 1])
with col2:
    st.title("Alphabet & Number Detection")

def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')

v_spacer(height=7, sb=False)


st.markdown (

    '''
    <style>
    [data-testid='stSidebar'][aria-expanded='true']>div:first-child{
        width:400px
    }

    [data-testid='stSidebar'][aria-expanded='false']>div:first-child{
        width:400px
        margin-left:-400px
    }
    </style>
    ''',
    unsafe_allow_html=True,
)


    
app_mode = st.sidebar.selectbox('Options',
['Number','Alphabet']
)

if app_mode =='Number':
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**Prediction**")
        kpi1_text = st.markdown("")

    with kpi2:
        st.markdown("**FrameRate**")
        kpi2_text = st.markdown("")

    st.markdown("<hr/>", unsafe_allow_html=True)
    prevTime = 0

    while run:
        _, frame = camera.read()
        frame, lmList = detectHand(frame)

        if len(lmList) != 0:
            fingers = []

            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)
            totalFingers = fingers.count(1)
            #print(totalFingers)

            # cv2.rectangle(frame, (0, 0), (90, 100), (0, 0, 0), cv2.FILLED)
            # cv2.putText(frame, str(totalFingers), (20, 80), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)
            
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{totalFingers}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)

        else:
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{0}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{0}</h1>", unsafe_allow_html=True)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    else:
        st.write('Stopped')



elif app_mode =='Alphabet':
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**Prediction**")
        kpi1_text = st.markdown("")

    with kpi2:
        st.markdown("**FrameRate**")
        kpi2_text = st.markdown("")

    st.markdown("<hr/>", unsafe_allow_html=True)
    prevTime = 0
    while run:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = camera.read()
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                    cx, cy = int(x * W), int(y * H)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)


                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))


            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            # cv2.rectangle(frame, (0, 0), (90, 100), (0, 0, 0), cv2.FILLED)
            # cv2.putText(frame, predicted_character, (20, 80) , cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)

            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{predicted_character}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)

        else:
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{0}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{0}</h1>", unsafe_allow_html=True)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    else:
        st.write('Stopped')


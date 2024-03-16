import cv2
import numpy as np
import time
import handModule as hd
import pyautogui as mouse

####################
wCam, hCam = 640, 480
frameR = 100
wScreen, hScreen = mouse.size()
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 7
####################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
print(wScreen, hScreen)
detector = hd.handDetector() 
while True:
    ret, img = cap.read()
    img = detector.findHands(img)

    lmlist, bbox = detector.findPos(img)
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        #print(x1, y1, x2, y2)

        fingers = detector.fingersUp()
        print(fingers)
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
        if fingers[1] == 1:
            if fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0: 
                mouse.moveTo(wScreen - clocX, clocY)
            elif fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                mouse.click(wScreen - clocX, clocY, 1, 0.1, "left")
            elif fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                mouse.scroll(25)
            elif fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0:
                mouse.click(wScreen - clocX, clocY, 1, 2, "right")
        else:
            if fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                mouse.scroll(-25)
        plocX, plocY = clocX, clocY
    # Fram Rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # Show image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
import mediapipe as mp
import cv2
import time
import math
import pyautogui as mouse

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelC = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.modelC,
                                        self.detectionCon, self.trackCon)
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, imgBGR):
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(imgBGR, hand, self.mp_hands.HAND_CONNECTIONS)
        
        return imgBGR
    
    def findPos(self, img):
        xList = []
        yList = []
        self.lmList = []
        bbox = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bbox = xMin, yMin, xMax, yMax
            #cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (0, 255, 0), 2)
            cv2.rectangle(img, (xMin-20, yMin-20), (xMax+20, yMax+20), (0, 255, 0), 2)
            x, y = self.lmList[8][1], self.lmList[8][2]
        return self.lmList, bbox
    
    def fingersUp(self):
        fingers = [1, 0, 0, 0, 0]
        for i in range(1, 5):
            if (self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i]-2][2]):
                fingers[i] = 1
            else:
                fingers[i] = 0
        return fingers


def main():
    pTime, cTime = 0, 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, img = cap.read()

        img = detector.findHands(img)
        lmList, bbox = detector.findPos(img)
        cv2.circle(img, (0, 100), 30, (250, 0, 250), 3)
        #detector.fingersUp(img)
        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()
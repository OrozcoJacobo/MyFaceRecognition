import threading

import cv2
from deepface import DeepFace

#What I want to do first in this case is to have a basic OpenCV camera structure
#So we want to find a camera, define the width and height 
#Go through endless loop until we terminate it, so we can capture a frame and do something with it 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)#I pass a 0 since I only have one camera, feel free to modify this in order to select a camera of your choosing

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


#This counter will be used to regulate the frequency with which a frame is tested for facial recognition, we don't want to waste tons of computational time testing every frame
counter = 0

#I also need something to keep track of wether the program found a match or not 
face_match = False

reference_img = cv2.imread("my_face.JPG")

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target = check_face, args =(frame.copy(), )).start()

            except ValueError:
                pass

        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
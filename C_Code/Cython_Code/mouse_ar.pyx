import mediapipe as mp
import cv2
import pyautogui
import time
import numpy as np
from deepface import DeepFace
import moveMouseOptimiced

aggiornareRiconoscimento = True
cdef int contaAggiornamenti = 0
faceMatch = False
referenceImg = cv2.imread("img_reference/IMG20240226084324~2.jpg")

# Disabilita failsafe di pyautogui
pyautogui.FAILSAFE = False

cdef int length_thumb_index = 0
cdef int ength_thumb_middle = 0
cdef int length_thumb_ring = 0
cdef int length_thumb_little = 0

mpHands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mpFaceMesh = mp.solutions.face_mesh

screenWidth, screenHeight = pyautogui.size()
draggingLeft = False 
draggingRight = False

gestureState = ""

# Avvia webcam
cap = cv2.VideoCapture(0)

pyautogui.PAUSE = 0
cdef list lmList = []

#cdef float start, end, totalTime
cdef int imageHeight, imageWidth
cdef int x, y
cdef int coordinateX, coordinateY
cdef int cx, cy

def checkFace(frame):
    global faceMatch
    
    try:
        if DeepFace.verify(frame, referenceImg.copy())['verified']:
            faceMatch = True
            
        else:
            faceMatch = False
            
    except ValueError:
        faceMatch = False

def calCoordinatesMouse(lmMouse):
    x = int(lmMouse.x * imageWidth)
    y = int(lmMouse.y * imageHeight)

    return (int(screenWidth / imageWidth * x), int(screenHeight / imageHeight * y))                   
                                                                                
with mpFaceMesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5
) as face:
    with mpHands.Hands(
        model_complexity = 1,
        max_num_hands = 2,
        min_detection_confidence = 0.2,
        min_tracking_confidence = 0.2) as hands:

        # Converti il formato del frame una sola volta
        while cap.isOpened():
            success, frame = cap.read()
            
            if aggiornareRiconoscimento and contaAggiornamenti == 0:
                checkFace(frame)
                contaAggiornamenti = 1
                aggiornareRiconoscimento = False
            
            start = time.time()
                        
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            
            frame.flags.writeable = False
            results = hands.process(frame)
            resultsFace = face.process(frame)
            frame.flags.writeable = True
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            imageHeight, imageWidth, _ = frame.shape
            
            """
            try:
                print("Prova")
               
                print("Prova 2")
            except ValueError:
                print("eRRORE")
                pass
            """
                        
            if faceMatch:
                cv2.putText(
                    frame,
                    "Access granted",
                    (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        
            else:
                cv2.putText(
                    frame,
                    "Denied access",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            if resultsFace.multi_face_landmarks:
                aggiornareRiconoscimento = True
                """for face_landmarks in resultsFace.multi_face_landmarks:
                    
                    mp_drawing.draw_landmarks(
                        image = frame,
                        landmark_list = face_landmarks, 
                        connections = mpFaceMesh.FACEMESH_TESSELATION,
                        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        landmark_drawing_spec = None
                    )
                    
                    mp_drawing.draw_landmarks(
                        image = frame,
                        landmark_list = face_landmarks, 
                        connections = mpFaceMesh.FACEMESH_IRISES,
                        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                        landmark_drawing_spec = None,
                    )"""

                if results.multi_hand_landmarks and faceMatch:
                    """for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks, 
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )"""

                    lmList.clear()
                    coordinateX, coordinateY = calCoordinatesMouse(results.multi_hand_landmarks[0].landmark[8])
                    moveMouseOptimiced.moveTo(coordinateX, coordinateY)

                    if len(results.multi_hand_landmarks) == 2:
                        for id, lm in enumerate(results.multi_hand_landmarks[1].landmark):
                            cx  = int(lm.x * imageWidth)
                            cy = int(lm.y * imageHeight)
                            lmList.append([id, cx, cy])
                        
                        # Calcola lunghezza solo se la lista non è vuota
                        if len(lmList) != 0:
                            lmArray = np.array(lmList)
                                                
                            length_thumb_middle = np.linalg.norm(lmArray[4][1:3] - lmArray[12][1:3])
                            length_thumb_index = np.linalg.norm(lmArray[4][1:3] - lmArray[8][1:3])
                            length_thumb_ring= np.linalg.norm(lmArray[4][1:3] - lmArray[16][1:3])
                            length_thumb_little= np.linalg.norm(lmArray[4][1:3] - lmArray[20][1:3])
                            
                            cv2.putText(frame, f'Gesture: {gestureState}', (lmArray[0][1], lmArray[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                            #cv2.rectangle(frame, (lmArray[9][1] - 100, lmArray[9][2] - 100), (lmArray[9][1] + 100, lmArray[9][2] + 100), (255, 0, 0), 2)

                        if length_thumb_index < 40 and not draggingLeft:
                            pyautogui.PAUSE = 0
                            pyautogui.mouseDown()
                            draggingLeft = True
                            gestureState = "Down Left Click"
                                                
                        elif length_thumb_index > 65 and draggingLeft:
                            pyautogui.mouseUp()
                            draggingLeft = False
                            gestureState = "Up Left Click"
                        
                        if length_thumb_middle < 40 and not draggingRight:
                            pyautogui.PAUSE = 0
                            pyautogui.mouseDown(button = "right")
                            draggingRight = True
                            gestureState = "Down Right Click"
                                                
                        elif length_thumb_middle > 65 and draggingRight:
                            pyautogui.mouseUp(button = "right")
                            draggingRight = False
                            gestureState = "Up Right Click"
                            #pyautogui.press('KEYTYPE_SOUND_DOWN')
                        
                        """ 
                        if fingers_up:
                            pyautogui.scroll(120)
                            print("Scrolling UP")

                        elif not fingers_up:
                            pyautogui.scroll(-120)
                            print("Scrolling DOWN")
                        """
            else: 
                contaAggiornamenti = 0
                aggiornareRiconoscimento = False

            fps = 1 / (time.time() - start)

            cv2.putText(
                frame, 
                f'FPS: {int(fps)}',
                (20,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.25, 
                (217, 85, 188), 
                2)       

            cv2.imshow('TrackMotion', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
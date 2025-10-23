import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Calculator buttons layout
buttons = [["7", "8", "9", "/"],
           ["4", "5", "6", "*"],
           ["1", "2", "3", "-"],
           ["0", ".", "=", "+"]]

button_positions = []
for i in range(len(buttons)):
    for j in range(len(buttons[i])):
        x = 80 * j + 20
        y = 80 * i + 20
        button_positions.append([x, y, buttons[i][j]])

# Calculator variables
expression = ""
result = ""

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb_frame)

    # Draw calculator buttons
    for pos in button_positions:
        x, y, text = pos
        cv2.rectangle(frame, (x, y), (x+70, y+70), (200, 200, 200), -1)
        cv2.putText(frame, text, (x+20, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Detect hands
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of index fingertip and thumb
            x1 = int(hand_landmarks.landmark[8].x * w)   # index tip
            y1 = int(hand_landmarks.landmark[8].y * h)
            x2 = int(hand_landmarks.landmark[4].x * w)   # thumb tip
            y2 = int(hand_landmarks.landmark[4].y * h)

            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)

            # Distance between index and thumb (pinch detection)
            distance = math.hypot(x2 - x1, y2 - y1)

            if distance < 40:  # pinch gesture
                for pos in button_positions:
                    x, y, text = pos
                    if x < x1 < x+70 and y < y1 < y+70:
                        cv2.rectangle(frame, (x, y), (x+70, y+70), (0, 255, 0), -1)
                        cv2.putText(frame, text, (x+20, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                        if text == "=":
                            try:
                                result = str(eval(expression))
                                expression = result
                            except:
                                result = "Error"
                                expression = ""
                        else:
                            expression += text

    # Display expression & result
    cv2.rectangle(frame, (20, 360), (340, 440), (0, 0, 0), -1)
    cv2.putText(frame, expression, (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Virtual Calculator", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

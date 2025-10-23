"""
gesture_calculator.py
Gesture-controlled on-screen calculator using OpenCV + MediaPipe.

Controls:
- Move index finger tip to hover buttons.
- Pinch (thumb tip + index tip close) to press a button.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ---- Configuration ----
CAM_WIDTH, CAM_HEIGHT = 1280, 720
PINCH_THRESHOLD = 0.045  # normalized distance threshold for pinch (tweak if needed)
DEBOUNCE_TIME = 0.4      # seconds between recognized clicks

# ---- UI / Button definitions ----
class Button:
    def __init__(self, x, y, w, h, label, fontsize=1.2):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.fontsize = fontsize
        self.is_hover = False
        self.is_pressed = False

    def draw(self, img):
        # background
        color_bg = (80, 80, 80)  # default bg
        if self.is_pressed:
            color_bg = (50, 160, 50)  # pressed (greenish)
        elif self.is_hover:
            color_bg = (70, 130, 180)  # hover (blueish)

        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), color_bg, -1)
        # border
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (30,30,30), 2)

        # label (centered)
        text_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, 2)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(img, self.label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, (255,255,255), 2)

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

# Build calculator buttons (simple layout)
def build_buttons(start_x, start_y, btn_w, btn_h, gap=10):
    labels = [
        ['7', '8', '9', '/'],
        ['4', '5', '6', '*'],
        ['1', '2', '3', '-'],
        ['0', '.', '=', '+'],
    ]
    buttons = []
    for r, row in enumerate(labels):
        for c, label in enumerate(row):
            x = start_x + c * (btn_w + gap)
            y = start_y + r * (btn_h + gap)
            b = Button(x, y, btn_w, btn_h, label)
            buttons.append(b)
    # add Clear and Backspace as separate buttons
    buttons.append(Button(start_x, start_y - (btn_h + gap), btn_w*2 + gap, btn_h, 'C'))      # clear
    buttons.append(Button(start_x + 2*(btn_w+gap), start_y - (btn_h + gap), btn_w*2 + gap, btn_h, '⌫'))  # backspace
    return buttons

# ---- Hand detection helper ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalized_to_pixel(norm_x, norm_y, w, h):
    px = min(int(norm_x * w), w-1)
    py = min(int(norm_y * h), h-1)
    return px, py

def normalized_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

# ---- Calculator logic ----
class Calculator:
    def __init__(self):
        self.expression = ""
        self.last_result = None

    def press(self, label):
        if label == 'C':
            self.expression = ""
        elif label == '⌫':
            self.expression = self.expression[:-1]
        elif label == '=':
            self.evaluate()
        else:
            # append digit/operator/dot
            # simple guard: don't allow two operators in a row (except minus)
            if len(self.expression) > 0 and self.expression[-1] in '+-*/' and label in '+*/':
                # ignore (prevents ++ or +*)
                return
            self.expression += label

    def evaluate(self):
        try:
            # safe-ish eval: allow digits, operators, dot
            # NOTE: eval can be dangerous if expression is untrusted; here we limit allowed chars.
            allowed = "0123456789+-*/.() "
            if all(ch in allowed for ch in self.expression):
                result = eval(self.expression)
                self.last_result = result
                self.expression = str(result)
            else:
                self.expression = "ERR"
        except Exception:
            self.expression = "ERR"

# ---- Main loop ----
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # construct UI
    btn_w, btn_h = 140, 100
    start_x = 40
    start_y = 220
    buttons = build_buttons(start_x, start_y, btn_w, btn_h, gap=18)
    calc = Calculator()

    last_click_time = 0

    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            results = hands.process(img_rgb)

            index_tip_px = None
            pinch = False

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                # get normalized coords of index tip (landmark 8) and thumb tip (landmark 4)
                lm_index = hand.landmark[8]
                lm_thumb = hand.landmark[4]
                index_tip_px = normalized_to_pixel(lm_index.x, lm_index.y, w, h)
                thumb_tip_px = (lm_thumb.x, lm_thumb.y)

                # compute normalized distance (relative to frame diagonal)
                norm_dist = normalized_distance((lm_index.x, lm_index.y), (lm_thumb.x, lm_thumb.y))
                pinch = norm_dist < PINCH_THRESHOLD

                # draw small circle at index fingertip
                cv2.circle(frame, index_tip_px, 8, (0, 255, 0), -1)

                # optional: draw hand skeleton (comment/uncomment)
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Reset hover/press states
            for b in buttons:
                b.is_hover = False
                b.is_pressed = False

            # if we have index finger coords, test hover and clicks
            if index_tip_px is not None:
                ix, iy = index_tip_px
                for b in buttons:
                    if b.contains(ix, iy):
                        b.is_hover = True
                        # click handling with debounce
                        if pinch:
                            now = time.time()
                            if now - last_click_time > DEBOUNCE_TIME:
                                b.is_pressed = True
                                calc.press(b.label)
                                last_click_time = now
                        break

            # Draw calculator panel background
            panel_w = start_x + 4*(btn_w+18) + 20
            panel_h = start_y + 4*(btn_h+18) + 40
            cv2.rectangle(frame, (start_x-20, 80), (panel_w, panel_h), (40,40,40), -1)
            cv2.rectangle(frame, (start_x-20, 80), (panel_w, panel_h), (20,20,20), 2)

            # draw display area
            display_x = start_x
            display_y = 90
            display_w = panel_w - start_x - 20
            display_h = 100
            cv2.rectangle(frame, (display_x, display_y), (display_x + display_w, display_y + display_h), (10,10,10), -1)
            cv2.rectangle(frame, (display_x, display_y), (display_x + display_w, display_y + display_h), (60,60,60), 2)

            # render expression text
            expr = calc.expression if calc.expression != "" else "0"
            # shrink text if too long
            font_scale = 2.0
            while cv2.getTextSize(expr, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0][0] > display_w - 30:
                font_scale -= 0.1
                if font_scale < 0.4:
                    break
            text_pos = (display_x + 20, display_y + display_h - 20)
            cv2.putText(frame, expr, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240,240,240), 3)

            # draw buttons
            for b in buttons:
                b.draw(frame)

            # show hint text
            cv2.putText(frame, "Pinch to Click | C: Clear | ⌫: Backspace | =: Evaluate", (start_x-20, panel_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Gesture Calculator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c') or key == ord('C'):  # keyboard fallback clear
                calc.press('C')
            elif key == ord('b') or key == ord('B'):  # keyboard fallback backspace
                calc.press('⌫')
            elif key == ord('=') or key == 13:  # Enter or '=' evaluate
                calc.press('=')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

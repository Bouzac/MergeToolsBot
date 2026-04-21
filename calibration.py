import cv2
import numpy as np
import mss
import time


MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

def nothing(x):
    pass


def main():
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("Top", "Trackbars", 0, 1080, nothing)
    cv2.createTrackbar("Left", "Trackbars", 0, 1920, nothing)
    cv2.createTrackbar("Bottom", "Trackbars", 1080, 1080, nothing)
    cv2.createTrackbar("Right", "Trackbars", 1920, 1920, nothing)

    with mss.mss() as sct:
        while True:
            time.sleep(0.05)

            screenshot = sct.grab(MONITOR)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            top = cv2.getTrackbarPos("Top", "Trackbars")
            left = cv2.getTrackbarPos("Left", "Trackbars")
            bot = cv2.getTrackbarPos("Bottom", "Trackbars")
            right = cv2.getTrackbarPos("Right", "Trackbars")

            croped_frame = frame[top:bot, left:right]

            cv2.imshow("Original vs Mask", croped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Crop final, top = {top}, left = {left}, bottom = {bot}, right = {right} ")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
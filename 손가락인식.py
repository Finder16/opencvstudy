import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i
    cnt = contours[ci]
    hull = cv2.convexHull(cnt)
    defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)

        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        angle = 0

        if b != 0 and c != 0:
            cos_angle = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
            angle = np.arccos(cos_angle)
            angle = angle * 180 / np.pi

            if angle <= 90:
                finger_count += 1
    if finger_count == 0:
        cv2.putText(frame, "0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    elif finger_count == 1:
        cv2.putText(frame, "1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    elif finger_count == 2:
        cv2.putText(frame, "2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    elif finger_count == 3:
        cv2.putText(frame, "3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    elif finger_count == 4:
        cv2.putText(frame, "4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

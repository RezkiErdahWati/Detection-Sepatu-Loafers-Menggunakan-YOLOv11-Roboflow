from ultralytics import YOLO
import cv2
import time

model = YOLO("best.pt")

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise Exception("No Camera")

while True:
    ret, image = cam.read()
    if not ret:
        break

    start = time.time()

    results = model(image)

    end = time.time()
    print("waktu:", end - start)

    frame = results[0].plot()

    cv2.imshow("Detection Loafers", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    print(f"Camera {i} ->", "OK" if ret else "FAIL")
    cap.release()

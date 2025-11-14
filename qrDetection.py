import cv2
import time

# Try AVFoundation backend on macOS for better camera support
cam = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Force camera output to 640x480 and remember the desired size
frame_size = (640, 480)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

qrDecoder = cv2.QRCodeDetector()

while cam.isOpened():
    loop_start = time.perf_counter()

    ret, frame = cam.read()
    if not ret or frame is None:
        print("Warning: Failed to grab frame from camera.")
        break

    # Resize incoming frames to the requested 640x480 window
    current_size = (frame.shape[1], frame.shape[0])
    if current_size != frame_size:
        frame = cv2.resize(frame, frame_size)

    value, points, _ = qrDecoder.detectAndDecode(frame)

    if value and points is not None:
        pts = points[0].astype(int)
        x1, y1 = pts[0]
        x2, y2 = pts[2]

        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x_center, y_center), 5, (255, 0, 0), -1)
        cv2.putText(
            frame,
            value,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"QR center: ({x_center}, {y_center})",
            (x_center, y_center + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    loop_end = time.perf_counter()
    frame_time = loop_end - loop_start
    fps = 1 / frame_time if frame_time > 0 else 0.0
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("QR Code Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

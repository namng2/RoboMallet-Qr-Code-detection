import cv2
import time
import numpy as np

# Choose the correct camera index (0, 1, 2...)
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)

if not cam.isOpened():
    print(f"Error: Could not open camera at index {CAM_INDEX}.")
    exit(1)

# Ask for a reasonable resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Confirm what we actually got
frame_size = (
    int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
print("Using resolution:", frame_size)


def detect_airpods(frame):
    """
    Very simple heuristic:
    - Find bright white-ish regions in HSV
    - Clean up with morphology
    - Keep medium-sized, tall-ish blobs (sort of like an AirPod stem)
    Returns list of bounding boxes (x, y, w, h)
    """
    # Blur a bit to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to HSV so we can isolate white
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # White-ish pixels: low saturation, high value
    # You will probably need to tune these numbers!
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:  # ignore tiny specks
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = h / float(w) if w > 0 else 0

        # Heuristics:
        # - Not ridiculously wide
        # - Tall-ish shape (stem-ish)
        # - Not giant (like a whole sheet of paper)
        if aspect_ratio < 1.2:  # too flat
            continue
        if area > (frame.shape[0] * frame.shape[1]) * 0.3:  # huge blob
            continue

        boxes.append((x, y, w, h))

    return boxes, mask


while cam.isOpened():
    loop_start = time.perf_counter()

    ret, frame = cam.read()
    if not ret or frame is None:
        print("Warning: Failed to grab frame from camera.")
        break

    boxes, mask = detect_airpods(frame)

    # Draw detection results
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cx, cy = x + w // 2, y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Object {i}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Center: ({cx}, {cy})",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    # FPS overlay
    loop_end = time.perf_counter()
    frame_time = loop_end - loop_start
    fps = 1 / frame_time if frame_time > 0 else 0.0

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Show both the frame and mask (for debugging)
    cv2.imshow("Object Detection", frame)
    cv2.imshow("White Mask (debug)", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

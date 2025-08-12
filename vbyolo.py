import cv2
import numpy as np
import mediapipe as mp
import sys
import os

# =============================
# Initialize MediaPipe Pose
# =============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def detect_volleyball(frame):
    """Detect volleyball by color and draw circle + label."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ----- Molten Ball Colors -----
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    molten_mask = cv2.bitwise_or(mask_red, mask_green)

    # ----- Mikasa Ball Colors -----
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mikasa_mask = cv2.bitwise_or(mask_blue, mask_yellow)

    # Combine both masks
    combined_mask = cv2.bitwise_or(molten_mask, mikasa_mask)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # filter out small noise
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, "Volleyball", (int(x - radius), int(y - radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"Volleyball detected at: {int(x)}, {int(y)}")
    return frame


def main():
    # Get video file from command line or default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "IMG_9969.MOV"  # default file name

    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: rotate frame if video is upside down
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Detect volleyball
        frame = detect_volleyball(frame)

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Example: print right wrist coordinates
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            print(f"Right wrist position: x={right_wrist.x:.2f}, y={right_wrist.y:.2f}")

        # Show result
        cv2.imshow('Volleyball Pose & Ball Detection', frame)

        # Exit on ESC
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

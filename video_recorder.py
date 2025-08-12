import cv2
import time

cap = cv2.VideoCapture(0)  # Adjust if using USB camera index or file

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

writer = cv2.VideoWriter(f'videos/output_{time.strftime("%Y%m%d_%H%M%S")}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

frame_idx = 0
while frame_idx < 100:  # Record 100 frames
    frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()
    if not ret:
        break

    writer.write(frame)

    # Display the frame (optional)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
writer.release()
# cv2.destroyAllWindows()
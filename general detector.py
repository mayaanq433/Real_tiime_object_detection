from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model (it can detect common objects including fruits)
model = YOLO("yolov8n.pt")  # You can also use 'yolov8s.pt' for better accuracy

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Fruit Detector", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
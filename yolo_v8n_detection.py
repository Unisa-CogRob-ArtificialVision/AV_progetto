import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "../test1.mp4"
cap = cv2.VideoCapture(video_path)

# Introuding frameskip
i = discard = 5
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        if (i % discard == 0):
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(source=frame, persist=True, tracker="botsort.yaml",classes=0)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            bbox = results.boxes

            # Display the annotated frame
            cv2.namedWindow('YOLOv8 Tracking', cv2.WINDOW_KEEPRATIO)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            cv2.resizeWindow('YOLOv8 Tracking', 720, 480) # custom resize, doesn't reflect the real ratio

        #increment the counter
        i += 1 

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
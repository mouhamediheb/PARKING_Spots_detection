import cv2
from util import get_Parking_spot_bboxes, empty_or_notempty

video_path = 'parking_1920_1080.mp4'
mask = cv2.imread('mask_1920_1080.png', 0)

# Get connected components from the mask
connected_components = cv2.connectedComponentsWithStats(mask, connectivity=4)

# Open video
cap = cv2.VideoCapture(video_path)
spots = get_Parking_spot_bboxes(connected_components)

# Initialize the status list
empty_stat_list = [None for j in spots]

frame_skip = 5  # Process every 10 frame
frame_count = 0

while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break

    # Skip frames: Process every 'frame_skip' frame
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Process the frame
    for spot_idx, spot in enumerate(spots):
        x, y, w, h = spot
        frame_crop = frame[y:y + h, x:x + w, :]

        # Check if the crop has valid dimensions (width and height > 0)
        if frame_crop.shape[0] == 0 or frame_crop.shape[1] == 0:
            continue  # Skip this spot if the crop is invalid or empty

        status_spot = empty_or_notempty(frame_crop)
        empty_stat_list[spot_idx] = status_spot

    # Now, use the updated status from empty_stat_list
    for spot_idx, spot in enumerate(spots):
        x, y, w, h = spot
        status_spot = empty_stat_list[spot_idx]  # Get the status from the list

        # Check the status and apply the corresponding color
        if status_spot == True:
            # If the spot is "not empty", draw a green rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green
        else:
            # If the spot is "empty", draw a red rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(empty_stat_list)), str(len(empty_stat_list))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.namedWindow('Parking Spot Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Parking Spot Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

# import cv2
# from distance_measuring import DistanceCalculator
# from object_detection import detect_objects, draw_detections

# # Known object size and camera focal length for distance calculation
# KNOWN_HEIGHT = 0.5  # Example: known height of a person (in meters)
# FOCAL_LENGTH = 600  # Example focal length

# # Initialize distance calculator
# distance_calculator = DistanceCalculator(FOCAL_LENGTH, KNOWN_HEIGHT)

# # Capture video from the webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Step 1: Detect objects in the current frame
#     detected_data, class_names = detect_objects(frame)

#     # Step 2: Draw detections and distances (if applicable) on the frame
#     frame_with_detections = draw_detections(frame, detected_data, class_names, distance_calculator)

#     # Display the frame with detections and distances
#     cv2.imshow('Object Detection with Distance Measurement', frame_with_detections)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close windows
# cap.release()
# cv2.destroyAllWindows()



import cv2
from object_detection import detect_objects, draw_detections
from distance_measuring import DistanceCalculator
from speech_output import speak_object_details  # Import the speech output module

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Example values for the camera's focal length and known object height (adjust these)
focal_length = 800  # Example value; this should be determined for your camera
known_height = 1.75  # Known height of the object (in meters), e.g., average human height

# Initialize the DistanceCalculator with these values
distance_calculator = DistanceCalculator(focal_length, known_height)

# Keep track of detected objects for speech output
spoken_objects = set()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects and their bounding boxes in the current frame
        detected_data, class_names = detect_objects(frame)

        # Debug: print detected object data
        print(f"Detected data: {detected_data}")

        # For each detected object, calculate the distance and display it
        frame_with_detections = draw_detections(frame, detected_data, class_names, distance_calculator)
        
        # Prepare a list of (object_name, distance) for speech output
        objects_with_distances = []
        for obj_data in detected_data:
            # Unpack obj_data manually (assuming it contains [x_center, y_center, width, height, confidence, class_id])
            try:
                x_center, y_center, width, height, confidence, class_id = obj_data
            except ValueError:
                print(f"Unexpected data format for obj_data: {obj_data}")
                continue  # Skip this detection if unpacking fails

            # Check if class_id is within the valid range of class_names
            class_id = int(class_id)  # Ensure it's an integer
            if class_id < len(class_names):
                object_name = class_names[class_id]
                
                # Calculate the pixel height as the height of the bounding box
                pixel_height = height

                # Calculate distance based on pixel height
                distance = distance_calculator.calculate_distance(pixel_height)
                objects_with_distances.append((object_name, distance))

                # Only speak the object name and distance once per entry into the frame
                if object_name not in spoken_objects:
                    print(f"Speaking: {object_name} at {distance:.2f} meters")  # Debug print
                    spoken_objects.add(object_name)  # Mark the object as spoken
                    speak_object_details([(object_name, distance)])  # Trigger speech output
            else:
                print(f"Warning: Detected class_id {class_id} is out of range for class_names.")

        # Display the frame with detected objects and distances
        cv2.imshow('Object Detection with Distance Measurement', frame_with_detections)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted! Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()

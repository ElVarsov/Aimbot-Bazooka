import cv2
import numpy as np
from ultralytics import YOLO
import time

# Distance calculation formula: Distance = (Focal length × Real object width) / Object width in pixels


# Change to your model path if needed. 
# Rasp pi 5 only supports nano models
MODEL_NAME = 'yolov8n.pt'  

# Camera parameters
FOCAL_LENGTH = 26     # Focal length in mm
SENSOR_WIDTH_MM = 26  # Sensor width in mm
REAL_OBJECT_WIDTH_MM = 4200
VIDEO_PATH = "25m1x.mov"  # Path to video file. Change to 0 (int) to use webcam
# CAMERA_OFFSET_MM = 0 # To calibrate the distance to the camera

# Lead targeting parameters
PROJECTILE_SPEED = 100  # Projectile speed in m/s 
MAX_HISTORY_FRAMES = 30  # Maximum number of past frames to keep for velocity calculation
TIME_WINDOW_VELOCITY = 30 # Actual time window for velocity calculations (frames)

velocity_history = []
accepted_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle'] # Classes we want to track

def calculate_focal_length_pixels(focal_length_camera_mm, image_width_pixels, sensor_width_mm):

    if focal_length_camera_mm <= 0 or image_width_pixels <= 0 or sensor_width_mm <= 0:
        raise ValueError("Focal length camera (mm), image width (px), and sensor width (mm) must be positive values.")
    focal_length_pixels = focal_length_camera_mm * image_width_pixels / sensor_width_mm
    return focal_length_pixels

def calculate_distance_mm(focal_length_pixels, real_object_width, object_size_in_pixels):

    if focal_length_pixels <= 0 or real_object_width <= 0 or object_size_in_pixels <= 0:
        raise ValueError("Focal length (px), real object width, and object size in pixels must be positive values.")
    distance = (focal_length_pixels * real_object_width) / object_size_in_pixels
    # distance -= CAMERA_OFFSET_MM 
    return distance 

def detect_objects(image, model):

    classes = model.names
    
    results = model.predict(image, verbose=False)

    detected_objects = []

    result = results[0]
    boxes = result.boxes

    for box in boxes:
        # Extract bounding box coordinates in (x1, y1, x2, y2) format
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Calculate box width and height
        w = x2 - x1
        h = y2 - y1

        # Calculate center coordinates of bounding box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Get class ID and confidence score
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        
        # Get class name from class ID
        label = classes[class_id]
        
        # Only track vehicle classes (cars, trucks, buses, motorcycles, bicycles)
        if label in accepted_classes:
            detected_objects.append({
                'class': label,
                'confidence': confidence,
                'w': w,
                'h': h,
                'center': (int(cx), int(cy)),
            })

    return detected_objects

def apply_affine_transformation(affine_matrix, point):

    dot_position_array = np.array([point], dtype=np.float32)
    A = affine_matrix[:, :2]
    t = affine_matrix[:, 2]
    transformed_point = (A @ dot_position_array.T + t.reshape(2, 1)).T
    transformed_dot_position_tuple = (int(transformed_point[0][0]), int(transformed_point[0][1]))
    return transformed_dot_position_tuple

def calculate_velocity(position_history, time_window = TIME_WINDOW_VELOCITY):

    if len(position_history) < 2:
        return (0, 0)  # Not enough data points to calculate velocity
        
    # Get the most recent positions
    recent_positions = position_history[-time_window:]

    
    # If we have enough history
    if len(recent_positions) >= 2:
        # Get start and end positions with timestamps
        start_pos, start_time = recent_positions[0]
        end_pos, end_time = recent_positions[-1]
        
        # Time difference in seconds
        time_diff = end_time - start_time
        
        if time_diff > 0:
            # Calculate velocity components (pixels per second)
            velocity_x = (end_pos[0] - start_pos[0]) / time_diff
            velocity_y = (end_pos[1] - start_pos[1]) / time_diff
            velocity_history.append((velocity_x, velocity_y))  # Store velocity for analysis
            return (velocity_x, velocity_y)
    
    return (0, 0)  # Default if calculation fails

def calculate_drop_off(distance_m, projectile_speed=PROJECTILE_SPEED):

    time_of_flight = distance_m / projectile_speed
    
    # Vertical drop due to gravity (in meters): y = 0.5 * g * t²
    # Where g is gravity (9.81 m/s^2)
    drop_m = 0.5 * 9.81 * (time_of_flight ** 2)
    
    return drop_m  # Return drop in meters (will be converted to pixels later)

def calculate_px_m_ratio(object_width_px, real_object_width_m):
    real_width_m = real_object_width_m / 1000.0
    return object_width_px / real_width_m

def predict_aim_point(current_position, velocity, distance_m, lead_time, object_width_px):

    # Simple linear prediction: new_position = current_position + velocity * time
    future_x = current_position[0] + velocity[0] * lead_time
    future_y = current_position[1] + velocity[1] * lead_time
    
    # Ensure the point stays within screen boundaries (assuming 1280x720 resolution)
    future_x = max(0, min(future_x, 1280))
    future_y = max(0, min(future_y, 720))

    # Calculate drop due to gravity (in meters)
    drop_m = calculate_drop_off(distance_m, PROJECTILE_SPEED)
    
    # Calculate pixels per meter for this target
    real_object_width_m = REAL_OBJECT_WIDTH_MM / 1000.0  # Convert mm to m
    px_per_meter = object_width_px / real_object_width_m
    
    # Convert drop from meters to pixels
    drop_px = drop_m * px_per_meter
    
    # Adjust aim point to compensate for gravity (aim higher)
    # Subtract because y-axis is inverted in image coordinates (0 at top)
    future_y -= drop_px
    
    return (int(future_x), int(future_y))

def create_velocity_graph(velocity_history):

    import matplotlib.pyplot as plt
    
    if not velocity_history:
        print("No velocity data to plot.")
        return
    
    velocities = np.array(velocity_history)
    time_steps = np.arange(len(velocities))
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, velocities[:, 0], label='Velocity X', color='blue')
    plt.plot(time_steps, velocities[:, 1], label='Velocity Y', color='red')
    
    plt.title('Velocity Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity (pixels/s)')
    plt.legend()
    plt.grid()
    plt.show()


def main():

    model = YOLO(MODEL_NAME)
    
    # Open the video capture device (file or webcam)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize ORB feature detector and matcher for frame-to-frame tracking
    orb = cv2.ORB_create(1000)  # Create ORB detector with 1000 features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Brute Force matcher with Hamming distance

    # Read the first frame
    # .read() returns a boolean indicating success and the frame itself
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        cap.release()
        exit()

    # Process first frame to get initial keypoints and descriptors
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    # Initialize tracking variables
    prev_points_queue = []  # Queue to store previous object center points
    tracking_history = {}   # Dictionary to store position history with timestamps
    
    frame_count = 0
    start_time = time.time()  # Record start time for timing calculations
    crosshair_prev_pos = None
    
    # Main loop for processing video frames
    while True:
        # Read a frame from the video source
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        # Calculate time since start for time-based calculations
        current_time = time.time() - start_time
        
        # Convert frame to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        # Skip processing if no features are found
        if des is None or prev_des is None:
            prev_gray = gray
            prev_kp, prev_des = kp, des
            continue

        # Match features between frames to detect camera movement
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)  # Sort by match quality
        good_matches = matches[:50]  # Use the top 50 best matches

        # Calculate the affine transformation between frames if enough good matches
        if len(good_matches) >= 3:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
            affine_matrix, inliers = cv2.estimateAffine2D(pts1, pts2)

        # Apply the affine transformation to all previously tracked points
        # This compensates for camera movement to keep tracking consistent
        prev_points_queue = list(map(lambda x: apply_affine_transformation(affine_matrix, x), prev_points_queue))

        if frame_count % 3 == 0:
        # Detect objects (vehicles) in the current frame
            detected_objects = detect_objects(frame, model)
            
            # Process the first detected object
            if detected_objects:
                main_object = detected_objects[0]  # Focus on the first detected object
                object_id = f"{main_object['class']}_{0}"  # Create a simple ID for the object
                current_position = main_object["center"]  # Center point of the object
                object_width_px = main_object["w"]  # Width of the object in pixels
                
                # Calculate distance to object using camera parameters
                focal_length_pixels = calculate_focal_length_pixels(FOCAL_LENGTH, frame.shape[1], SENSOR_WIDTH_MM)
                distance_mm = calculate_distance_mm(focal_length_pixels, REAL_OBJECT_WIDTH_MM, object_width_px)
                distance_m = distance_mm / 1000  # Convert to meters
                
                # Store position with timestamp for velocity calculation
                if object_id not in tracking_history:
                    tracking_history[object_id] = []
                tracking_history[object_id].append((current_position, current_time))
                
                # Limit history size to prevent using too much memory
                if len(tracking_history[object_id]) > MAX_HISTORY_FRAMES:
                    tracking_history[object_id].pop(0)  # Remove oldest entry
                
                # Calculate velocity from position history
                velocity = calculate_velocity(tracking_history[object_id])
                
                # Calculate lead time based on distance and projectile speed    
                # Lead time = distance / speed (basic physics formula)
                lead_time = distance_m / PROJECTILE_SPEED  # Time in seconds
                
                # Predict where to aim based on object movement
                aim_point = predict_aim_point(current_position, velocity, distance_m, lead_time, object_width_px)
                
                crosshair_prev_pos = aim_point
                                
                # Calculate drop for display purposes
                drop_m = calculate_drop_off(distance_m, PROJECTILE_SPEED)
                real_object_width_m = REAL_OBJECT_WIDTH_MM / 1000.0
                px_per_meter = object_width_px / real_object_width_m
                drop_px = drop_m * px_per_meter
                
                # Display distance and drop information at the bottom of the screen
                info_text = f"Distance: {distance_m:.1f}m | Drop: {drop_m:.2f}m ({drop_px:.1f}px)"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add current position to tracking queue
                prev_points_queue.append(current_position)

        
        cv2.drawMarker(frame, crosshair_prev_pos, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Limit tracking queue size to prevent cluttering the visualization
        if len(prev_points_queue) > 5:
            prev_points_queue.pop(0)  # Remove oldest point

        # Display the processed frame
        cv2.imshow("Vehicle Tracking with Lead Targeting", frame)

        # Update previous frame information for next iteration
        prev_gray = gray
        prev_kp, prev_des = kp, des
        prev_frame = frame.copy()
        frame_count += 1

        # Check for user pressing 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # If you want to see the velocity graph, uncomment this line
    # create_velocity_graph(velocity_history)


# Entry point of the script
if __name__ == "__main__":
    main()
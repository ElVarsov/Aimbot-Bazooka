import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import hailo
import math
import time

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.FOCAL_LENGTH = 26     # Focal length in mm
        self.SENSOR_WIDTH_MM = 26  # Sensor width in mm
        self.REAL_OBJECT_WIDTH_MM = 4200  # Real width of target object in mm  
        self.RESOLUTION = (640, 480)  # Resolution of the video

        # Targeting parameters
        self.MAX_HISTORY_FRAMES = 30  # Maxqimum number of past frames to keep for velocity calculation
        self.TIME_WINDOW_VELOCITY = 30  # Actual time window for velocity calculations (frames)
        self.DETECTION_INTERVAL = 1  # Interval for object detection (in frames)
        self.CROSSHAIR_POS = (self.RESOLUTION[0] // 2, self.RESOLUTION[1] // 2)  # position in the center
        self.MAX_BULLET_TRAVEL_TIME = 5

        # Physics constants
        self.GRAVITY = 9.81  # m/s^2
        self.AIR_DENSITY = 1.225  # kg/m³ at sea level

        # Airsoft BB configuration
        self.AIRSOFT_CONFIG = {
            "muzzle_velocity": 115,  # m/s (380 ft/s)
            "bb_mass": 0.00025,  # kg (0.25g BB)
            "bb_diameter": 0.006,  # m (6mm)
            "drag_coefficient": 0.47,  # Sphere drag coefficient
        }

        self.velocity_history = []   
        self.accepted_classes = ["car"]   # Classes we want to track
        self.prev_points_queue = []  # Queue to store previous object center points
        self.tracking_history = {}

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.affine_matrix = None
        self.new_aim_point = None
        self.crosshair_prev_pos = self.CROSSHAIR_POS
        
        # Initialize ORB detector and matcher
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Frame counter for detection interval
        self.frame_counter = 0

    def calculate_focal_length_pixels(self, focal_length_camera_mm, image_width_pixels, sensor_width_mm):
        if focal_length_camera_mm <= 0 or image_width_pixels <= 0 or sensor_width_mm <= 0:
            raise ValueError("Focal length camera (mm), image width (px), and sensor width (mm) must be positive values.")
        focal_length_pixels = focal_length_camera_mm * image_width_pixels / sensor_width_mm
        return focal_length_pixels
    
    def calculate_distance_mm(self, focal_length_pixels, real_object_width, object_size_in_pixels):
        if focal_length_pixels <= 0 or real_object_width <= 0 or object_size_in_pixels <= 0:
            raise ValueError("Focal length (px), real object width, and object size in pixels must be positive values.")
        distance = (focal_length_pixels * real_object_width) / object_size_in_pixels
        return distance 

    def detect_objects(self, image):

        detected_objects = []
        roi = hailo.get_roi_from_buffer(image)
        boxes = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        for box in boxes:

            x1, y1, x2, y2 = box.bbox()
            w = (x2 - x1) * self.RESOLUTION[0]
            h = (y2 - y1) * self.RESOLUTION[1]
            cx = (x1 + x2) / 2 * self.RESOLUTION[0]
            cy = (y1 + y2) / 2 * self.RESOLUTION[1]
            label = box.get_label()
            conf = box.get_confidence()
            
            if label in self.accepted_classes:
                detected_objects.append({
                    'class': label,
                    'w': w,
                    'h': h,
                    'confidence': conf, 
                    'center': (int(cx), int(cy)),
                })

        return detected_objects

    def apply_affine_transformation(self, affine_matrix, point):
        dot_position_array = np.array([point], dtype=np.float32)
        A = affine_matrix[:, :2]
        t = affine_matrix[:, 2]
        transformed_point = (A @ dot_position_array.T + t.reshape(2, 1)).T
        transformed_dot_position_tuple = (int(transformed_point[0][0]), int(transformed_point[0][1]))
        return transformed_dot_position_tuple

    def calculate_ballistic_trajectory_simple(self, distance_m, config):
        """
        Calculate bullet trajectory using simple drag and Euler's method
        Returns drop in meters and time of flight
        """
        muzzle_velocity = config["muzzle_velocity"]  # m/s
        mass = config["bb_mass"]  # kg
        diameter = config["bb_diameter"]  # m
        drag_coeff = config["drag_coefficient"]
        
        # Calculate cross-sectional area
        area = math.pi * (diameter / 2) ** 2  # m²
        
        # Time step for numerical integration (seconds)
        dt = 0.001
        
        # Initial conditions
        velocity_x = muzzle_velocity  # m/s horizontal
        velocity_y = 0  # m/s vertical (starts horizontal)
        x = 0  # horizontal position (m)
        y = 0  # vertical position (m) - positive is up
        
        time_of_flight = 0
        
        # Numerical integration using Euler's method
        while x < distance_m and time_of_flight < self.MAX_BULLET_TRAVEL_TIME:
            # Calculate total velocity magnitude
            velocity_magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
            
            if velocity_magnitude <= 0:
                break
                
            # Calculate drag force magnitude
            drag_force = 0.5 * self.AIR_DENSITY * drag_coeff * area * velocity_magnitude**2
            
            # Calculate drag acceleration components (opposite to velocity direction)
            drag_accel_x = -(drag_force / mass) * (velocity_x / velocity_magnitude)
            drag_accel_y = -(drag_force / mass) * (velocity_y / velocity_magnitude)
            
            # Update velocities (drag + gravity)
            velocity_x += drag_accel_x * dt
            velocity_y += (drag_accel_y - self.GRAVITY) * dt  # Gravity acts downward
            
            # Update positions
            x += velocity_x * dt
            y += velocity_y * dt
            
            time_of_flight += dt
            
            # Break if velocity becomes too low
            if velocity_magnitude < 5:  # m/s
                break
        
        # Return drop in meters (negative y means drop) and time of flight
        drop_m = abs(y) if y < 0 else 0
        
        return drop_m, time_of_flight

    def calculate_flight_time(self, distance_m, config):
        """
        Calculate time of flight using simple ballistic model
        """
        _, time_of_flight = self.calculate_ballistic_trajectory_simple(distance_m, config)
        
        if time_of_flight > 0 and time_of_flight < self.MAX_BULLET_TRAVEL_TIME:
            print(f"Time of flight: {time_of_flight:.3f} seconds, Distance: {distance_m:.1f} m")
            return time_of_flight
        else:
            print(f"Failed to fetch flight time., Distance:{distance_m:.1f} m]")
            return 0

    def calculate_velocity(self, position_history):
        if len(position_history) < 2:
            return (0, 0)  # Not enough data points to calculate velocity
            
        time_window = 30  # seconds
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
                self.velocity_history.append((velocity_x, velocity_y))  # Store velocity for analysis
                return (velocity_x, velocity_y)
        
        return (0, 0)  # Default if calculation fails

    def calculate_drop_off(self, distance_m, config):
        """
        Calculate bullet drop using simple ballistic model
        Returns drop in meters
        """
        drop_m, _ = self.calculate_ballistic_trajectory_simple(distance_m, config)
        return drop_m

    def calculate_px_m_ratio(self, object_width_px, real_object_width_m):
        real_width_m = real_object_width_m / 1000.0
        return object_width_px / real_width_m

    def predict_aim_point(self, current_position, velocity, distance_m, lead_time, object_width_px, config):
        # Simple linear prediction: new_position = current_position + velocity * time
        future_x = current_position[0] + velocity[0] * lead_time
        future_y = current_position[1] + velocity[1] * lead_time

        print(f"Current X: {current_position[0]}, Future X: {future_x}")
        
        # Ensure the point stays within screen boundaries
        future_x = max(0, min(future_x, self.RESOLUTION[0]))
        future_y = max(0, min(future_y, self.RESOLUTION[1]))

        # Calculate drop using simple ballistic model
        drop_m = self.calculate_drop_off(distance_m, config)
        
        # Calculate pixels per meter for this target
        real_object_width_m = self.REAL_OBJECT_WIDTH_MM / 1000.0  # Convert mm to m
        px_per_meter = object_width_px / real_object_width_m
        
        # Convert drop from meters to pixels
        drop_px = drop_m * px_per_meter
        
        # Adjust aim point to compensate for gravity (aim higher)
        # Subtract because y-axis is inverted in image coordinates (0 at top)
        future_y -= drop_px
        
        return (int(future_x), int(future_y))

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    start_time = time.time()
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    frame_count = user_data.get_count()

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Skip processing if frame is None
    if frame is None:
        return Gst.PadProbeReturn.OK

    current_time = time.time() - start_time

    # Draw crosshair
    cv2.drawMarker(frame, user_data.CROSSHAIR_POS, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

    # Convert to grayscale for feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = user_data.orb.detectAndCompute(gray, None)

    # Initialize previous frame data on first run
    if user_data.prev_gray is None:
        user_data.prev_gray = gray
        user_data.prev_kp = kp
        user_data.prev_des = des
        return Gst.PadProbeReturn.OK

    # Skip processing if no features are found
    if des is None or user_data.prev_des is None:
        user_data.prev_gray = gray
        user_data.prev_kp, user_data.prev_des = kp, des
        return Gst.PadProbeReturn.OK

    # Match features between frames to detect camera movement
    matches = user_data.bf.match(user_data.prev_des, des)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by match quality
    good_matches = matches[:50]  # Use the top 50 best matches

    # Calculate the affine transformation between frames if enough good matches
    if len(good_matches) >= 3:
        pts1 = np.float32([user_data.prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
        user_data.affine_matrix, inliers = cv2.estimateAffine2D(pts1, pts2)

    # Apply the affine transformation to all previously tracked points
    if user_data.affine_matrix is not None:
        user_data.prev_points_queue = [user_data.apply_affine_transformation(user_data.affine_matrix, x) for x in user_data.prev_points_queue]
    
    # Process detections at specified interval
    if user_data.frame_counter % user_data.DETECTION_INTERVAL == 0:
        detections = user_data.detect_objects(buffer)  # Pass buffer instead of frame

        if detections:
            main_obj = detections[0]
            object_id = f"{main_obj['class']}_{0}"
            
            current_position = main_obj["center"]  # Center point of the object
            object_width_px = main_obj["w"]  # Width of the object in pixels
            
            # Calculate distance to object using camera parameters
            focal_length_pixels = user_data.calculate_focal_length_pixels(user_data.FOCAL_LENGTH, user_data.RESOLUTION[0], user_data.SENSOR_WIDTH_MM)
            distance_mm = user_data.calculate_distance_mm(focal_length_pixels, user_data.REAL_OBJECT_WIDTH_MM, object_width_px)
            distance_m = distance_mm / 1000  # Convert to meters
            
            # Store position with timestamp for velocity calculation
            if object_id not in user_data.tracking_history:
                user_data.tracking_history[object_id] = []
            user_data.tracking_history[object_id].append((current_position, current_time))
            
            # Limit history size to prevent using too much memory
            if len(user_data.tracking_history[object_id]) > user_data.MAX_HISTORY_FRAMES:
                user_data.tracking_history[object_id].pop(0)  # Remove oldest entry
            
            # Calculate velocity from position history
            velocity = user_data.calculate_velocity(user_data.tracking_history[object_id])
            
            # Calculate lead time using simple ballistic model
            lead_time = user_data.calculate_flight_time(distance_m, user_data.AIRSOFT_CONFIG)  # Time in seconds
            
            # Predict where to aim based on object movement
            aim_point = user_data.predict_aim_point(current_position, velocity, distance_m, lead_time, object_width_px, user_data.AIRSOFT_CONFIG)

            offset_x = aim_point[0] - current_position[0]
            offset_y = abs(aim_point[1] - current_position[1])
            
            user_data.crosshair_prev_pos = aim_point

            user_data.new_aim_point = (user_data.CROSSHAIR_POS[0] - offset_x, int(user_data.CROSSHAIR_POS[1] + offset_y))
                            
            # Calculate drop for display purposes
            drop_m = user_data.calculate_drop_off(distance_m, user_data.AIRSOFT_CONFIG)
            real_object_width_m = user_data.REAL_OBJECT_WIDTH_MM / 1000.0
            px_per_meter = object_width_px / real_object_width_m
            drop_px = drop_m * px_per_meter
            
            # Display distance and drop information at the bottom of the screen
            info_text = f"Distance: {distance_m:.1f}m | Drop: {drop_m:.3f}m ({drop_px:.1f}px) | Time: {lead_time:.3f}s"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add current position to tracking queue
            user_data.prev_points_queue.append(current_position)
    
    # Draw the adjusted aim point
    aim_marker_pos = user_data.new_aim_point if user_data.new_aim_point else user_data.CROSSHAIR_POS
    cv2.drawMarker(frame, aim_marker_pos, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # Update previous frame information
    user_data.prev_gray = gray
    user_data.prev_kp = kp
    user_data.prev_des = des
    user_data.frame_counter += 1

    # Limit tracking queue size
    if len(user_data.prev_points_queue) > 5:
        user_data.prev_points_queue.pop(0)

    # Set the frame for display
    if user_data.use_frame:
        # Convert the frame from RGB to BGR format for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
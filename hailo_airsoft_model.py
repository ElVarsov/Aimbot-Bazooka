import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo
from hailo_apps_infra.hailo_rpi_common import app_callback_class
from hailo_apps_infra.detection_pipeline_simple import GStreamerDetectionApp
import cv2
import numpy as np
import time
import math

# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'yolov8n.pt'  

        # Camera parameters
        self.FOCAL_LENGTH = 26     # Focal length in mm
        self.SENSOR_WIDTH_MM = 26  # Sensor width in mm
        self.REAL_OBJECT_WIDTH_MM = 4200  # Real width of target object in mm
        self.VIDEO_PATH = "25m1x.mov"  # Path to video file. Change to 0 (int) to use webcam
        self.RESOLUTION = (640, 640)  # Resolution of the video

        # Targeting parameters
        self.MAX_HISTORY_FRAMES = 30  # Maximum number of past frames to keep for velocity calculation
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
        self.accepted_classes = ["car"]  # Classes we want to track
        
        # Tracking variables
        self.tracking_history = {}  # Dictionary to store position history with timestamps
        self.start_time = time.time()
        self.new_aim_point = None
        self.crosshair_prev_pos = self.CROSSHAIR_POS
        
        # ORB feature detector for camera movement compensation
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_points_queue = []

        print("Airsoft Ballistic Tracking System Initialized")
        print(f"Muzzle velocity: {self.AIRSOFT_CONFIG['muzzle_velocity']} m/s")
        print(f"BB mass: {self.AIRSOFT_CONFIG['bb_mass']*1000:.2f} g")
        print(f"BB diameter: {self.AIRSOFT_CONFIG['bb_diameter']*1000:.0f} mm")
        print(f"Drag coefficient: {self.AIRSOFT_CONFIG['drag_coefficient']}")

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
            print(f"No valid solution for flight time, target may be too far., Distance:{distance_m:.1f} m]")
            return 0

    def calculate_velocity(self, position_history, time_window=None):
        if time_window is None:
            time_window = self.TIME_WINDOW_VELOCITY
            
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

    def process_detections(self, detections, frame_data=None):
        """
        Process Hailo detections and perform ballistic calculations
        """
        current_time = time.time() - self.start_time
        detected_objects = []
        
        # Convert Hailo detections to our format
        for detection in detections:
            label = detection.get_label()
            if label in self.accepted_classes:
                bbox = detection.get_bbox()
                confidence = detection.get_confidence()
                
                # Calculate center and dimensions
                x1, y1, x2, y2 = bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                detected_objects.append({
                    'class': label,
                    'confidence': confidence,
                    'w': w,
                    'h': h,
                    'center': (int(cx), int(cy)),
                })
        
        # Process the first detected object for ballistic calculations
        if detected_objects:
            main_object = detected_objects[0]  # Focus on the first detected object
            object_id = f"{main_object['class']}_{0}"  # Create a simple ID for the object
            current_position = main_object["center"]  # Center point of the object
            object_width_px = main_object["w"]  # Width of the object in pixels
            
            # Calculate distance to object using camera parameters
            focal_length_pixels = self.calculate_focal_length_pixels(
                self.FOCAL_LENGTH, self.RESOLUTION[0], self.SENSOR_WIDTH_MM
            )
            distance_mm = self.calculate_distance_mm(
                focal_length_pixels, self.REAL_OBJECT_WIDTH_MM, object_width_px
            )
            distance_m = distance_mm / 1000  # Convert to meters
            
            # Store position with timestamp for velocity calculation
            if object_id not in self.tracking_history:
                self.tracking_history[object_id] = []
            self.tracking_history[object_id].append((current_position, current_time))
            
            # Limit history size to prevent using too much memory
            if len(self.tracking_history[object_id]) > self.MAX_HISTORY_FRAMES:
                self.tracking_history[object_id].pop(0)  # Remove oldest entry
            
            # Calculate velocity from position history
            velocity = self.calculate_velocity(self.tracking_history[object_id])
            
            # Calculate lead time using simple ballistic model
            lead_time = self.calculate_flight_time(distance_m, self.AIRSOFT_CONFIG)  # Time in seconds
            
            # Predict where to aim based on object movement
            aim_point = self.predict_aim_point(
                current_position, velocity, distance_m, lead_time, object_width_px, self.AIRSOFT_CONFIG
            )

            offset_x = aim_point[0] - current_position[0]
            offset_y = abs(aim_point[1] - current_position[1])
            
            self.crosshair_prev_pos = aim_point
            self.new_aim_point = (self.CROSSHAIR_POS[0] - offset_x, int(self.CROSSHAIR_POS[1] + offset_y))
                            
            # Calculate drop for display purposes
            drop_m = self.calculate_drop_off(distance_m, self.AIRSOFT_CONFIG)
            real_object_width_m = self.REAL_OBJECT_WIDTH_MM / 1000.0
            px_per_meter = object_width_px / real_object_width_m
            drop_px = drop_m * px_per_meter
            
            # Store info for display
            self.ballistic_info = {
                'distance_m': distance_m,
                'drop_m': drop_m,
                'drop_px': drop_px,
                'lead_time': lead_time,
                'velocity': velocity,
                'current_position': current_position
            }
            
            return True
        
        return False

    def create_velocity_graph(self):
        """
        Create velocity graph if matplotlib is available
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.velocity_history:
                print("No velocity data to plot.")
                return
            
            velocities = np.array(self.velocity_history)
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
        except ImportError:
            print("Matplotlib not available for velocity graph.")


# User-defined callback function: This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    user_data.increment()  # Using the user_data to count the number of frames
    
    buffer = info.get_buffer()  # Get the GstBuffer from the probe info
    if buffer is None:  # Check if the buffer is valid
        return Gst.PadProbeReturn.OK
    
    roi = hailo.get_roi_from_buffer(buffer)  # Get the ROI from the buffer
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)  # Get the detections from the buffer

    # Build detection string for display
    string_to_print = f"Frame count: {user_data.get_count()}\n"
    
    # Process detections for ballistic calculations
    if detections:
        has_target = user_data.process_detections(detections)
        
        # Display detection info
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            string_to_print += f"Detection: {label} Confidence: {confidence:.2f}\n"
            
            # Add ballistic info if this is our target class
            if label in user_data.accepted_classes and hasattr(user_data, 'ballistic_info'):
                info = user_data.ballistic_info
                string_to_print += f"Distance: {info['distance_m']:.1f}m | "
                string_to_print += f"Drop: {info['drop_m']:.3f}m ({info['drop_px']:.1f}px) | "
                string_to_print += f"Time: {info['lead_time']:.3f}s\n"
                string_to_print += f"Velocity: X={info['velocity'][0]:.1f} Y={info['velocity'][1]:.1f} px/s\n"
                
                if user_data.new_aim_point:
                    string_to_print += f"Aim Point: {user_data.new_aim_point}\n"
    else:
        string_to_print += "No detections\n"
    
    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()  # Create an instance of the user app callback class
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
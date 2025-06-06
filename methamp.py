import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import threading
import queue
from collections import deque
import psutil
import os
import smbus
from scipy.spatial.transform import Rotation

# Performance optimizations for Raspberry Pi 5
try:
    os.nice(-5)  # Higher priority
    psutil.Process().cpu_affinity([0, 1, 2])  # Use 3 cores
except:
    pass

accepted_targets = ["car", "person"]  # YOLO classes to track

# Optimized parameters for Pi 5
MODEL_NAME = 'yolov8n.pt'
FOCAL_LENGTH = 3
SENSOR_WIDTH_MM = 3
REAL_OBJECT_WIDTH_MM = 450
VIDEO_PATH = 0

# Optimized resolutions
DISPLAY_RESOLUTION = (640, 480)
YOLO_RESOLUTION = (640, 480)  # Lower resolution for YOLO processing

USE_GRAYSCALE = True

# Video recording settings
SAVE_VIDEO = True  # Set to False to disable recording
OUTPUT_FILENAME = "ballistic_tracker_output.mp4"  # Output filename
OUTPUT_FPS = 30  # Output video FPS
OUTPUT_CODEC = 'mp4v'  # Codec (alternatives: 'XVID', 'MJPG', 'mp4v')

# Threading parameters
MAX_QUEUE_SIZE = 15  # Small queue to reduce latency
DETECTION_INTERVAL = 1  # Process every 3rd frame
MAX_HISTORY_FRAMES = 15  # Reduced from 30

# Some affine tranformation type shi
ENABLE_AFFINE_COMPENSATION = True  # Enable/disable camera movement compensation
ORB_FEATURES = 500  # Reduced from 1000 for better performance
MIN_GOOD_MATCHES = 10  # Minimum matches needed for affine calculation
MAX_GOOD_MATCHES = 30  # Maximum matches to use (for performance)


# Physics constants
GRAVITY = 9.81
AIRSOFT_CONFIG = {
    "muzzle_velocity": 115,
    "bb_mass": 0.00025,
    "bb_diameter": 0.006,
    "drag_coefficient": 0.47,
}

# IMU6050 (MPU6050) Configuration
IMU_I2C_ADDRESS = 0x68
IMU_I2C_BUS = 1
IMU_SAMPLE_RATE_HZ = 100  # Hz
IMU_FILTER_ALPHA = 0.98  # Complementary filter coefficient

# Pre-calculate constants
FOCAL_LENGTH_PIXELS = FOCAL_LENGTH * DISPLAY_RESOLUTION[0] / SENSOR_WIDTH_MM
CROSSHAIR_POS = (DISPLAY_RESOLUTION[0] // 2, DISPLAY_RESOLUTION[1] // 2)

class IMU6050:
    """IMU6050 (MPU6050) sensor interface for Raspberry Pi"""
    
    def __init__(self, bus_number=IMU_I2C_BUS, address=IMU_I2C_ADDRESS):
        self.bus = smbus.SMBus(bus_number)
        self.address = address
        self.gyro_offset = {'x': 0, 'y': 0, 'z': 0}
        self.accel_offset = {'x': 0, 'y': 0, 'z': 0}
        
        # Orientation state
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.last_time = time.time()
        
        self.initialize_sensor()
        self.calibrate()
    
    def initialize_sensor(self):
        """Initialize the MPU6050 sensor"""
        try:
            # Wake up the MPU6050
            self.bus.write_byte_data(self.address, 0x6B, 0x00)
            
            # Set accelerometer range to ±2g
            self.bus.write_byte_data(self.address, 0x1C, 0x00)
            
            # Set gyroscope range to ±250°/s
            self.bus.write_byte_data(self.address, 0x1B, 0x00)
            
            # Set sample rate divider (1kHz / (1 + 7) = 125Hz)
            self.bus.write_byte_data(self.address, 0x19, 0x07)
            
            # Configure digital low pass filter
            self.bus.write_byte_data(self.address, 0x1A, 0x06)
            
            print("IMU6050 initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize IMU6050: {e}")
            raise
    
    def read_word_2c(self, reg):
        """Read 16-bit signed value from register"""
        val = self.bus.read_word_data(self.address, reg)
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val
    
    def read_sensor_data(self):
        """Read raw accelerometer and gyroscope data"""
        try:
            # Read accelerometer data
            accel_x = self.read_word_2c(0x3B) / 16384.0  # ±2g range
            accel_y = self.read_word_2c(0x3D) / 16384.0
            accel_z = self.read_word_2c(0x3F) / 16384.0
            
            # Read gyroscope data
            gyro_x = (self.read_word_2c(0x43) / 131.0) - self.gyro_offset['x']  # ±250°/s range
            gyro_y = (self.read_word_2c(0x45) / 131.0) - self.gyro_offset['y']
            gyro_z = (self.read_word_2c(0x47) / 131.0) - self.gyro_offset['z']
            
            # Apply accelerometer offset
            accel_x -= self.accel_offset['x']
            accel_y -= self.accel_offset['y']
            accel_z -= self.accel_offset['z']
            
            return {
                'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
                'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z}
            }
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            return None
    
    def calibrate(self, samples=1000):
        """Calibrate the sensor by finding offsets"""
        print("Calibrating IMU6050... Keep device still!")
        
        gyro_sum = {'x': 0, 'y': 0, 'z': 0}
        accel_sum = {'x': 0, 'y': 0, 'z': 0}
        
        for i in range(samples):
            data = self.read_sensor_data()
            if data:
                gyro_sum['x'] += data['gyro']['x']
                gyro_sum['y'] += data['gyro']['y']
                gyro_sum['z'] += data['gyro']['z']
                
                accel_sum['x'] += data['accel']['x']
                accel_sum['y'] += data['accel']['y']
                accel_sum['z'] += data['accel']['z'] - 1.0  # Account for gravity
            
            if i % 100 == 0:
                print(f"Calibration progress: {i/samples*100:.1f}%")
            
            time.sleep(0.005)  # 200Hz sampling during calibration
        
        # Calculate offsets
        self.gyro_offset['x'] = gyro_sum['x'] / samples
        self.gyro_offset['y'] = gyro_sum['y'] / samples
        self.gyro_offset['z'] = gyro_sum['z'] / samples
        
        self.accel_offset['x'] = accel_sum['x'] / samples
        self.accel_offset['y'] = accel_sum['y'] / samples
        self.accel_offset['z'] = accel_sum['z'] / samples
        
        print(f"Calibration complete!")
        print(f"Gyro offsets: {self.gyro_offset}")
        print(f"Accel offsets: {self.accel_offset}")
    
    def update_orientation(self, data):
        """Update orientation using complementary filter"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt > 0.1:  # Skip if time delta is too large (first reading)
            return
        
        # Get sensor data
        accel = data['accel']
        gyro = data['gyro']
        
        # Calculate pitch and roll from accelerometer
        accel_pitch = math.atan2(accel['y'], math.sqrt(accel['x']**2 + accel['z']**2)) * 180 / math.pi
        accel_roll = math.atan2(-accel['x'], accel['z']) * 180 / math.pi
        
        # Integrate gyroscope data
        self.pitch += gyro['x'] * dt
        self.roll += gyro['y'] * dt
        self.yaw += gyro['z'] * dt
        
        # Apply complementary filter
        self.pitch = IMU_FILTER_ALPHA * self.pitch + (1 - IMU_FILTER_ALPHA) * accel_pitch
        self.roll = IMU_FILTER_ALPHA * self.roll + (1 - IMU_FILTER_ALPHA) * accel_roll
        
        # Keep yaw in range [-180, 180]
        if self.yaw > 180:
            self.yaw -= 360
        elif self.yaw < -180:
            self.yaw += 360
    
    def get_orientation(self):
        """Get current orientation in degrees"""
        return {
            'pitch': self.pitch,
            'roll': self.roll,
            'yaw': self.yaw
        }

class BallisticTracker:
    def __init__(self):
        self.model = YOLO(MODEL_NAME)
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.tracking_history = {}
        self.velocity_history = deque(maxlen=100)
        self.current_aim_point = CROSSHAIR_POS
        self.running = True
        self.flip_180 = False
        self.video_writer = None
        
        # IMU initialization
        self.imu_data_queue = queue.Queue(maxsize=50)
        self.current_orientation = {'pitch': 0, 'roll': 0, 'yaw': 0}
        self.imu = None
        self.init_imu()

        if ENABLE_AFFINE_COMPENSATION:
            self.orb = cv2.ORB_create(ORB_FEATURES)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.prev_kp = None
            self.prev_des = None
            self.prev_points_queue = deque(maxlen=10)  # Track recent object positions
        
    def apply_affine_transformation(self, affine_matrix, point):
        """Apply affine transformation to a point"""
        if affine_matrix is None:
            return point
            
        dot_position_array = np.array([point], dtype=np.float32)
        A = affine_matrix[:, :2]
        t = affine_matrix[:, 2]
        transformed_point = (A @ dot_position_array.T + t.reshape(2, 1)).T
        transformed_dot_position_tuple = (int(transformed_point[0][0]), int(transformed_point[0][1]))
        return transformed_dot_position_tuple
    
    def calculate_affine_transformation(self, gray_frame):
        """Calculate affine transformation between current and previous frame"""
        if not ENABLE_AFFINE_COMPENSATION:
            return None
            
        # Detect keypoints and descriptors
        kp, des = self.orb.detectAndCompute(gray_frame, None)
        
        if des is None or self.prev_des is None:
            self.prev_kp, self.prev_des = kp, des
            return None
        
        # Match features between frames
        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:MAX_GOOD_MATCHES]  # Use top matches
        
        affine_matrix = None
        if len(good_matches) >= MIN_GOOD_MATCHES:
            try:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                affine_matrix, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            except:
                affine_matrix = None
        
        # Update previous frame data
        self.prev_kp, self.prev_des = kp, des
        return affine_matrix
    
    def init_imu(self):
        """Initialize IMU sensor"""
        try:
            self.imu = IMU6050()
            print("IMU6050 initialized successfully")
        except Exception as e:
            print(f"Failed to initialize IMU6050: {e}")
            print("Continuing without IMU...")
            self.imu = None
    
    def imu_thread(self):
        """IMU reading thread - runs at high frequency"""
        if not self.imu:
            return
            
        target_interval = 1.0 / IMU_SAMPLE_RATE_HZ
        
        while self.running:
            try:
                start_time = time.time()
                
                # Read sensor data
                data = self.imu.read_sensor_data()
                if data:
                    # Update orientation
                    self.imu.update_orientation(data)
                    
                    # Get current orientation
                    orientation = self.imu.get_orientation()
                    
                    # Update shared orientation data (thread-safe)
                    if not self.imu_data_queue.full():
                        self.imu_data_queue.put(orientation)
                
                # Maintain consistent sampling rate
                elapsed = time.time() - start_time
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"IMU thread error: {e}")
                time.sleep(0.1)
                continue
    
    def get_latest_imu_data(self):
        """Get the latest IMU orientation data"""
        try:
            # Get the most recent orientation data
            while not self.imu_data_queue.empty():
                self.current_orientation = self.imu_data_queue.get_nowait()
        except queue.Empty:
            pass
        
        return self.current_orientation
    
    def apply_orientation_compensation(self, aim_point, orientation):
        """Apply IMU-based orientation compensation to aim point"""
        # Convert degrees to radians
        pitch_rad = math.radians(orientation['pitch'])
        roll_rad = math.radians(orientation['roll'])
        
        # Calculate compensation offsets (simplified approach)
        # These values may need tuning based on your setup
        pitch_compensation = math.tan(pitch_rad) * 100  # pixels per degree approximation
        roll_compensation = math.tan(roll_rad) * 100
        
        compensated_x = aim_point[0] + roll_compensation
        compensated_y = aim_point[1] - pitch_compensation  # Negative because screen Y is inverted
        
        # Clamp to screen boundaries
        compensated_x = max(0, min(compensated_x, DISPLAY_RESOLUTION[0]))
        compensated_y = max(0, min(compensated_y, DISPLAY_RESOLUTION[1]))
        
        return (int(compensated_x), int(compensated_y))
        
    def calculate_distance_mm(self, object_size_px):
        """Optimized distance calculation with pre-calculated focal length"""
        if object_size_px <= 0:
            return 0
        return (FOCAL_LENGTH_PIXELS * REAL_OBJECT_WIDTH_MM) / object_size_px
    
    def calculate_ballistic_simple(self, distance_m):
        """Simplified ballistic calculation with larger timestep"""
        if distance_m <= 0:
            return 0, 0
            
        muzzle_velocity = AIRSOFT_CONFIG["muzzle_velocity"]
        mass = AIRSOFT_CONFIG["bb_mass"]
        diameter = AIRSOFT_CONFIG["bb_diameter"]
        drag_coeff = AIRSOFT_CONFIG["drag_coefficient"]
        
        area = math.pi * (diameter / 2) ** 2
        dt = 0.01  # Larger timestep for performance
        
        velocity_x = muzzle_velocity
        velocity_y = 0
        x, y = 0, 0
        time_of_flight = 0
        
        while x < distance_m and time_of_flight < 3:  # Reduced max time
            velocity_magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
            if velocity_magnitude <= 5: 
                break
                
            drag_force = 0.5 * 1.225 * drag_coeff * area * velocity_magnitude**2
            drag_accel_x = -(drag_force / mass) * (velocity_x / velocity_magnitude)
            drag_accel_y = -(drag_force / mass) * (velocity_y / velocity_magnitude)
            
            velocity_x += drag_accel_x * dt
            velocity_y += (drag_accel_y - GRAVITY) * dt
            
            x += velocity_x * dt
            y += velocity_y * dt
            time_of_flight += dt
        
        drop_m = abs(y) if y < 0 else 0
        return drop_m, time_of_flight
    
    def detect_objects_thread(self):
        """YOLO detection thread - runs independently"""
        frame_count = 0
        
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                frame, timestamp = frame_data
                frame_count += 1
                
                # Only process every Nth frame
                if frame_count % DETECTION_INTERVAL != 0:
                    continue
                
                # Resize frame for YOLO processing
                yolo_frame = cv2.resize(frame, YOLO_RESOLUTION)
                
                if USE_GRAYSCALE:
                    if len(yolo_frame.shape) == 3:
                        gray_frame = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2GRAY)
                        yolo_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                # Run YOLO detection
                results = self.model.predict(yolo_frame, verbose=False)
                detected_objects = []
                
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            class_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())
                            label = self.model.names[class_id]
                            
                            if label in accepted_targets and confidence > 0.5:
                                # Scale coordinates back to display resolution
                                scale_x = DISPLAY_RESOLUTION[0] / YOLO_RESOLUTION[0]
                                scale_y = DISPLAY_RESOLUTION[1] / YOLO_RESOLUTION[1]
                                
                                w = (x2 - x1) * scale_x
                                h = (y2 - y1) * scale_y
                                cx = (x1 + x2) / 2 * scale_x
                                cy = (y1 + y2) / 2 * scale_y
                                
                                detected_objects.append({
                                    'class': label,
                                    'confidence': confidence,
                                    'w': w,
                                    'h': h,
                                    'center': (int(cx), int(cy)),
                                    'bbox': (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
                                })
                
                # Send results back
                if not self.result_queue.full():
                    self.result_queue.put((detected_objects, timestamp))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection thread error: {e}")
                continue
    
    def calculate_velocity(self, position_history, time_window=7):
        """Simplified velocity calculation"""
        if len(position_history) < 2:
            return (0, 0)
            
        # Convert deque to list for slicing, or use list() conversion
        recent_positions = list(position_history)[-time_window:]
        if len(recent_positions) >= 2:
            start_pos, start_time = recent_positions[0]
            end_pos, end_time = recent_positions[-1]
            
            time_diff = end_time - start_time
            if time_diff > 0:
                velocity_x = (end_pos[0] - start_pos[0]) / time_diff
                velocity_y = (end_pos[1] - start_pos[1]) / time_diff
                return (velocity_x, velocity_y)
        
        return (0, 0)
    
    def predict_aim_point(self, current_position, velocity, distance_m, lead_time, object_width_px):
        """Calculate aim point with ballistic compensation"""
        future_x = current_position[0] + velocity[0] * lead_time
        future_y = current_position[1] + velocity[1] * lead_time
        
        # Calculate drop compensation
        drop_m, _ = self.calculate_ballistic_simple(distance_m)
        real_object_width_m = REAL_OBJECT_WIDTH_MM / 1000.0
        px_per_meter = object_width_px / real_object_width_m
        drop_px = drop_m * px_per_meter
        
        # Compensate for drop (aim higher)
        future_y -= drop_px
        
        # Clamp to screen boundaries
        future_x = max(0, min(future_x, DISPLAY_RESOLUTION[0]))
        future_y = max(0, min(future_y, DISPLAY_RESOLUTION[1]))
        
        return (int(future_x), int(future_y))
    
    def setup_video_writer(self, fps):
        """Initialize video writer for recording"""
        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
            self.video_writer = cv2.VideoWriter(
                OUTPUT_FILENAME,
                fourcc,
                OUTPUT_FPS,  # Use consistent output FPS
                DISPLAY_RESOLUTION
            )
            if self.video_writer.isOpened():
                print(f"Recording video to: {OUTPUT_FILENAME}")
                print(f"Output FPS: {OUTPUT_FPS}, Codec: {OUTPUT_CODEC}")
            else:
                print("Failed to open video writer!")
                self.video_writer = None
    
    def cleanup_video_writer(self):
        """Clean up video writer"""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved to: {OUTPUT_FILENAME}")
    
    def main_loop(self):
        """Main video processing loop"""
        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_RESOLUTION[1])
        
        # Get video FPS for timing control
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:  # Fallback if FPS detection fails
            video_fps = 30
        frame_duration = 1.0 / video_fps  # Time between frames in seconds
        
        print(f"Video FPS: {video_fps}, Frame duration: {frame_duration:.4f}s")
        
        # Setup video recording
        self.setup_video_writer(video_fps)
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detect_objects_thread, daemon=True)
        detection_thread.start()
        
        # Start IMU thread
        if self.imu:
            imu_thread = threading.Thread(target=self.imu_thread, daemon=True)
            imu_thread.start()
            print("IMU thread started")
        
        start_time = time.time()
        frame_count = 0
        last_frame_time = start_time
        

        if ENABLE_AFFINE_COMPENSATION:
            ret, prev_frame = cap.read()
            if ret:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                self.prev_kp, self.prev_des = self.orb.detectAndCompute(prev_gray, None)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = time.time() - start_time
                frame_count += 1
                
                affine_matrix = None
                if ENABLE_AFFINE_COMPENSATION:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    affine_matrix = self.calculate_affine_transformation(gray_frame)
                    
                    # Apply affine transformation to previous tracked points
                    if affine_matrix is not None and len(self.prev_points_queue) > 0:
                        transformed_points = []
                        for point in self.prev_points_queue:
                            transformed_point = self.apply_affine_transformation(affine_matrix, point)
                            transformed_points.append(transformed_point)
                        self.prev_points_queue = deque(transformed_points, maxlen=10)
                

                # Get latest IMU data
                orientation = self.get_latest_imu_data()
                
                # Send frame to detection thread (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), current_time))
                
                # Check for detection results
                try:
                    detected_objects, detection_time = self.result_queue.get_nowait()
                    
                    if detected_objects:
                        main_object = detected_objects[0]
                        object_id = f"{main_object['class']}_{0}"
                        current_position = main_object["center"]
                        object_width_px = main_object["w"]
                        
                        if ENABLE_AFFINE_COMPENSATION:
                            self.prev_points_queue.append(current_position)
                        # Calculate distance
                        distance_mm = self.calculate_distance_mm(object_width_px)
                        distance_m = distance_mm / 1000
                        
                        # Update tracking history
                        if object_id not in self.tracking_history:
                            self.tracking_history[object_id] = deque(maxlen=MAX_HISTORY_FRAMES)
                        self.tracking_history[object_id].append((current_position, detection_time))
                        
                        # Calculate velocity and aim point
                        velocity = self.calculate_velocity(self.tracking_history[object_id])
                        _, lead_time = self.calculate_ballistic_simple(distance_m)
                        
                        if lead_time > 0:
                            aim_point = self.predict_aim_point(
                                current_position, velocity, distance_m, lead_time, object_width_px
                            )
                            
                            # Apply IMU-based orientation compensation
                            if self.imu:
                                aim_point = self.apply_orientation_compensation(aim_point, orientation)
                            
                            # Update crosshair position
                            offset_x = aim_point[0] - current_position[0]
                            offset_y = aim_point[1] - current_position[1]
                            
                            self.current_aim_point = (CROSSHAIR_POS[0] - offset_x, CROSSHAIR_POS[1] + abs(offset_y)) if not self.flip_180 else (CROSSHAIR_POS[0] - offset_x, CROSSHAIR_POS[1] - abs(offset_y))
                            
                            # Display info
                            drop_m, _ = self.calculate_ballistic_simple(distance_m)
                            info_text = f"Distance: {distance_m:.1f}m | Drop: {drop_m:.3f}m | Time: {lead_time:.3f}s"
                            cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                except queue.Empty:
                    pass
                
                # Draw crosshairs
                cv2.drawMarker(frame, CROSSHAIR_POS, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.drawMarker(frame, self.current_aim_point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                # Display IMU orientation data
                if self.imu:
                    imu_text = f"Pitch: {orientation['pitch']:.1f}° | Roll: {orientation['roll']:.1f}° | Yaw: {orientation['yaw']:.1f}°"
                    cv2.putText(frame, imu_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Show actual video FPS vs processing FPS
                processing_fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"Video FPS: {video_fps:.1f} | Processing: {processing_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add recording indicator
                if SAVE_VIDEO and self.video_writer is not None:
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 8, (0, 0, 255), -1)  # red dot
                
                if self.flip_180:
                    frame = cv2.flip(frame, -1)

                # Write frame to video file before displaying
                if SAVE_VIDEO and self.video_writer is not None:
                    self.video_writer.write(frame)

                cv2.imshow("Optimized Ballistic Tracker with IMU", frame)
                
                # Control playback timing to match video FPS
                current_time = time.time()
                elapsed_since_last_frame = current_time - last_frame_time
                time_to_wait = frame_duration - elapsed_since_last_frame
                
                if time_to_wait > 0:
                    # Use cv2.waitKey with calculated delay
                    wait_ms = max(1, int(time_to_wait * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                else:
                    # If we're running behind, just check for key press
                    key = cv2.waitKey(1) & 0xFF
                
                last_frame_time = current_time
                
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup
            self.running = False
            cap.release()
            self.cleanup_video_writer()  # Save and close video file
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BallisticTracker()
    tracker.main_loop()
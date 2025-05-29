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

# some optimazations for the rpi5
try:
    os.nice(-5)  # higher priority
    psutil.Process().cpu_affinity([0, 1, 2])  # use 3 cores
except:
    pass

accepted_targets = ["car"]  

# Optimized parameters for Pi 5
MODEL_NAME = 'yolov8n.pt'
FOCAL_LENGTH = 70
SENSOR_WIDTH_MM = 8
REAL_OBJECT_WIDTH_MM = 4200
VIDEO_PATH = 0

# Optimized resolutions
DISPLAY_RESOLUTION = (800, 480)
YOLO_RESOLUTION = (640, 480)  

# Video recording settings
SAVE_VIDEO = False  
OUTPUT_FILENAME = "ballistic_tracker_output.mp4"
OUTPUT_FPS = 30  
OUTPUT_CODEC = 'mp4v'  #alternatives: 'XVID', 'MJPG', 'mp4v'

USE_GRAYSCALE = True

# Threading parameters
MAX_QUEUE_SIZE = 15
DETECTION_INTERVAL = 1 
MAX_HISTORY_FRAMES = 15 

# Affine transformation settings
ENABLE_AFFINE_COMPENSATION = True
ORB_FEATURES = 500  
MIN_GOOD_MATCHES = 10  
MAX_GOOD_MATCHES = 30  

# Physics constants
GRAVITY = 9.81
AIRSOFT_CONFIG = {
    "muzzle_velocity": 115,
    "bb_mass": 0.00025,
    "bb_diameter": 0.006,
    "drag_coefficient": 0.47,
}

FOCAL_LENGTH_PIXELS = FOCAL_LENGTH * DISPLAY_RESOLUTION[0] / SENSOR_WIDTH_MM
CROSSHAIR_POS = (640, 360)
#CROSSHAIR_POS = (DISPLAY_RESOLUTION[0] // 2, DISPLAY_RESOLUTION[1] // 2)  # Center of the display resolution

MAX_FLIGHT_TIME = 5  # seconds, reduced for performance
CONF_THRESHOLD = 0.5  # Confidence threshold for detections

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
        
        # Affine transformation variables
        if ENABLE_AFFINE_COMPENSATION:
            self.orb = cv2.ORB_create(ORB_FEATURES)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.prev_kp = None
            self.prev_des = None
            self.prev_points_queue = deque(maxlen=10)
        
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
            
        keypoints, des = self.orb.detectAndCompute(gray_frame, None)
        
        if des is None or self.prev_des is None:
            self.prev_kp, self.prev_des = keypoints, des
            return None
        
        # match features between frames
        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:MAX_GOOD_MATCHES]  # use top matches
        
        affine_matrix = None
        if len(good_matches) >= MIN_GOOD_MATCHES:
            try:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
                affine_matrix, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            except:
                affine_matrix = None
        
        # update previous frame data
        self.prev_kp, self.prev_des = keypoints, des
        return affine_matrix
        
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
        dt = 0.01  # increase timestep to improve performance
        
        velocity_x = muzzle_velocity
        velocity_y = 0
        x, y = 0, 0
        time_of_flight = 0
        
        while x < distance_m and time_of_flight < MAX_FLIGHT_TIME:
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
        """YOLO detection thread - runs independently with grayscale processing"""
        frame_count = 0
        
        while self.running:
            try:
                # get frame from queue (non-blocking)
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                frame, timestamp = frame_data
                frame_count += 1
                
                if frame_count % DETECTION_INTERVAL != 0:
                    continue
                
                # resize for yolo
                yolo_frame = cv2.resize(frame, YOLO_RESOLUTION)
                
                # convert to grayscale for better performance if enabled
                if USE_GRAYSCALE:
                    if len(yolo_frame.shape) == 3:  # If it's a color frame
                        gray_frame = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2GRAY)
                        # Convert back to 3-channel for YOLO
                        yolo_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

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
                            
                            if label in accepted_targets and confidence > CONF_THRESHOLD:
                    
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
                
                # send results back
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
            
        # convert deque to list for slicing, or use list() conversion
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
        
        drop_m, _ = self.calculate_ballistic_simple(distance_m)
        real_object_width_m = REAL_OBJECT_WIDTH_MM / 1000.0
        px_per_meter = object_width_px / real_object_width_m
        drop_px = drop_m * px_per_meter
        
        future_y -= drop_px
        
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
        
        # get video FPS for timing control
        video_fps = 30
        if video_fps <= 0:  # fallback if FPS detection fails
            video_fps = 30
        frame_duration = 1.0 / video_fps  # time between frames in seconds
        
        print(f"Video FPS: {video_fps}, Frame duration: {frame_duration:.4f}s")
        print(f"Grayscale processing: {'Enabled' if USE_GRAYSCALE else 'Disabled'}")
        print(f"Affine compensation: {'Enabled' if ENABLE_AFFINE_COMPENSATION else 'Disabled'}")
        
        self.setup_video_writer(video_fps)
        
        detection_thread = threading.Thread(target=self.detect_objects_thread, daemon=True)
        detection_thread.start()
        
        start_time = time.time()
        frame_count = 0
        last_frame_time = start_time
        
        # initialize first frame for affine transformation
        if ENABLE_AFFINE_COMPENSATION:
            ret, prev_frame = cap.read()
            if ret:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                self.prev_kp, self.prev_des = self.orb.detectAndCompute(prev_gray, None)
        
        try:
            c_frames_not_detected = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = time.time() - start_time
                frame_count += 1
                
                cv2.drawMarker(frame, CROSSHAIR_POS, (255, 0, 0), cv2.MARKER_CROSS, 1300, 2)

                # calculate affine transformation for camera movement compensation
                affine_matrix = None
                if ENABLE_AFFINE_COMPENSATION:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    affine_matrix = self.calculate_affine_transformation(gray_frame)
                    
                    # apply affine transformation to previous tracked points
                    if affine_matrix is not None and len(self.prev_points_queue) > 0:
                        transformed_points = []
                        for point in self.prev_points_queue:
                            transformed_point = self.apply_affine_transformation(affine_matrix, point)
                            transformed_points.append(transformed_point)
                        self.prev_points_queue = deque(transformed_points, maxlen=10)
                
                # send frame to detection thread (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), current_time))

                try:
                    detected_objects, detection_time = self.result_queue.get_nowait()
                    
                    if detected_objects:
                        c_frames_not_detected = 0
                        main_object = detected_objects[0]
                        object_id = f"{main_object['class']}_{0}"
                        current_position = main_object["center"]
                        object_width_px = main_object["w"]

                        if ENABLE_AFFINE_COMPENSATION:
                            self.prev_points_queue.append(current_position)

                        distance_mm = self.calculate_distance_mm(object_width_px)
                        distance_m = distance_mm / 1000
                        
                        # update tracking history
                        if object_id not in self.tracking_history:
                            self.tracking_history[object_id] = deque(maxlen=MAX_HISTORY_FRAMES)
                        self.tracking_history[object_id].append((current_position, detection_time))
                        
                        velocity = self.calculate_velocity(self.tracking_history[object_id])
                        _, lead_time = self.calculate_ballistic_simple(distance_m)
                        
                        if lead_time > 0:
                            aim_point = self.predict_aim_point(
                                current_position, velocity, distance_m, lead_time, object_width_px
                            )

                            offset_x = aim_point[0] - current_position[0]
                            offset_y = aim_point[1] - current_position[1]
                            
                            self.current_aim_point = (CROSSHAIR_POS[0] - offset_x, CROSSHAIR_POS[1] + abs(offset_y)) if not self.flip_180 else (CROSSHAIR_POS[0] - offset_x, CROSSHAIR_POS[1] - abs(offset_y))
                            
                            # info display
                            drop_m, _ = self.calculate_ballistic_simple(distance_m)
                            compensation_status = "ON" if ENABLE_AFFINE_COMPENSATION and affine_matrix is not None else "OFF"
                            info_text = f"Distance: {distance_m:.1f}m | Drop: {drop_m:.3f}m | Time: {lead_time:.3f}s | Stab: {compensation_status}"
                            cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        c_frames_not_detected += 1 
            

                except queue.Empty:
                    pass

                if c_frames_not_detected <=  5:
                    cv2.drawMarker(frame, self.current_aim_point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                # Draw tracking points if affine compensation is enabled
                '''
                THIS DRAWS THE PREVIOUS POSITIONS OF THE OBJECTS
                ATTENTION: BIG DELAY

                if ENABLE_AFFINE_COMPENSATION and len(self.prev_points_queue) > 0:
                    for i, point in enumerate(self.prev_points_queue):
                        color_intensity = int(255 * (i + 1) / len(self.prev_points_queue))
                        cv2.circle(frame, point, 3, (0, color_intensity, 255 - color_intensity), -1)
                
                '''        
                # show actual video FPS vs processing FPS
                processing_fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"Video FPS: {video_fps:.1f} | Processing: {processing_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # add recording indicator
                if SAVE_VIDEO and self.video_writer is not None:
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 8, (0, 0, 255), -1)  # red dot
                
                if self.flip_180:
                    frame = cv2.flip(frame, -1)

                if SAVE_VIDEO and self.video_writer is not None:
                    self.video_writer.write(frame)

                cv2.imshow("Enhanced Ballistic Tracker", frame)
                
                # control playback timing to match video FPS
                current_time = time.time()
                elapsed_since_last_frame = current_time - last_frame_time
                time_to_wait = frame_duration - elapsed_since_last_frame
                
                if time_to_wait > 0:
                    # use cv2.waitKey with calculated delay
                    wait_ms = max(1, int(time_to_wait * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                else:
                    # if we're running behind, just check for key press
                    key = cv2.waitKey(1) & 0xFF
                
                last_frame_time = current_time
                
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # cleanup
            self.running = False
            cap.release()
            self.cleanup_video_writer()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BallisticTracker()
    tracker.main_loop()
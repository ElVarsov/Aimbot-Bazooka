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

# Threading parameters
MAX_QUEUE_SIZE = 15  # Small queue to reduce latency
DETECTION_INTERVAL = 1  # Process every 3rd frame
MAX_HISTORY_FRAMES = 15  # Reduced from 30

# Physics constants
GRAVITY = 9.81
AIRSOFT_CONFIG = {
    "muzzle_velocity": 115,
    "bb_mass": 0.00025,
    "bb_diameter": 0.006,
    "drag_coefficient": 0.47,
}

# Pre-calculate constants
FOCAL_LENGTH_PIXELS = FOCAL_LENGTH * DISPLAY_RESOLUTION[0] / SENSOR_WIDTH_MM
CROSSHAIR_POS = (DISPLAY_RESOLUTION[0] // 2, DISPLAY_RESOLUTION[1] // 2)

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
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detect_objects_thread, daemon=True)
        detection_thread.start()
        
        start_time = time.time()
        frame_count = 0
        last_frame_time = start_time
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time() - start_time
            frame_count += 1
            
            # Send frame to detection thread (non-blocking)
            if not self.frame_queue.full():
                self.frame_queue.put((frame.copy(), current_time))
            
            # Check for detection results
            try:
                detected_objects, detection_time = self.result_queue.get_nowait()
                
                if detected_objects:
                    '''
                    for obj in detected_objects:
                        if "bbox" in obj:
                            x1, y1, x2, y2 = obj['bbox']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label_text = f"{obj['class']}: {obj['confidence']:.2f}, "
                            cv2.putText(frame, label_text, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    '''        
                    main_object = detected_objects[0]
                    object_id = f"{main_object['class']}_{0}"
                    current_position = main_object["center"]
                    object_width_px = main_object["w"]
                    
                    # Calculate distance
                    distance_mm = self.calculate_distance_mm(object_width_px)
                    distance_m = distance_mm / 1000
                    print(f"Distance: {distance_m:.1f}m")
                    
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
            
            # Show actual video FPS vs processing FPS
            processing_fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"Video FPS: {video_fps:.1f} | Processing: {processing_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.flip_180:
                frame = cv2.flip(frame, -1)

            cv2.imshow("Optimized Ballistic Tracker", frame)
            
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
        
        self.running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BallisticTracker()
    tracker.main_loop()
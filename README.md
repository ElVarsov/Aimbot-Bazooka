# Aimbot-Bazooka
Sako hackaton


Some important calculations for estimating the crosshair:

1. Focal length in pixels:

focal_length_px = (focal_length_camera_mm * image_width_pixels) / sensor_width_mm

2. Distance calculations:

distance_mm = (focal_length_pixels * real_object_width_mm) / object_width_in_pixels

3. Velocity calculations:

velocity_x = (end_pos[0] - start_pos[0]) / (end_time - start_time)
velocity_y = (end_pos[1] - start_pos[1]) / (end_time - start_time)

4. Lead time (how long it takes for the bullet to reach its target)

lead_time = distance_m / PROJECTILE_SPEED

5. Predicted future position (basically the aiming point / red crosshair):

future_x = current_position[0] + velocity[0] * lead_time
future_y = current_position[1] + velocity[1] * lead_time



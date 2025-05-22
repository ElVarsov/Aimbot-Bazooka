# Aimbot-Bazooka
Sako hackaton

G1 BALLISTICS CALCULATIONS CHECKLIST

1. DETERMINE DISTANCE FROM VEHICLE

Distance = Focal length(px) * Real object width (mm) / Object size (px)

2. CONVERT DISTANCE FROM METERS TO YARDS (to comply with g1 standards)

distance_yards = distance_m * 1.09361

3. USE G1 MODEL TO COMPUTE DROP AND TIME

functions:
    - calculate_ballistic_trajectory(distance_yards, projectile_config)
    - g1_drag_coefficient(mach_number)

steps:
    - Initialize physical conditions
        - Muzzle velocity
        - Position x=0, y=0
        - Time step dt = 0.001
        - Speed of sound constant: 1116.4 ft/s

    - Simulate projectile flight with Euler's method
        - Loop until total horizontal distance x >= target distance
        - At each step do:
            - Compute mach number: mach = velocity / speed of sound
            - Get c_drag drag coefficient from g1_drag_coefficient(mach)
            - Apply drag (deceleration): a_drag = c_drag * v_old^2 / (b_c * 1000)    // b_c is ballistic coefficient 
            - Update velocity: new_v = old_v - a_drag * dt
            - Update horizontal position: x_new = x_old + v_new * dt

    - When projectile reaches target
        - Record vertical displacement y (drop)
        - Conversion from feet -> inches -> meters / pixels
        - Store flight time

4. USE DROP AND TIME FOR PREDICTING AIM POINT

drop_m = calculate_drop_off_g1(distance_m, projectile_config)
lead_time = calculate_flight_time_g1(distance_m, projectile_config)




# Aimbot-Bazooka
**Real-Time Ballistic Trajectory Prediction System** | Sako Hackathon Project

## Overview

An advanced computer vision system that combines YOLOv8 object detection with physics-based ballistic calculations to predict optimal aim points for military applications. Features multi-threaded processing, camera stabilization, and hardware acceleration (HailoAI) support for deployment on Raspberry Pi 5.

## Key Features

- **Real-Time Object Detection**: YOLOv8n model for target identification
- **Ballistic Physics Engine**: Accounts for gravity, air resistance, and projectile characteristics
- **Motion Prediction**: Velocity estimation and lead calculation for moving targets
- **Camera Stabilization**: ORB feature-based motion compensation
- **Multi-Platform Support**: PC, Raspberry Pi 5, and Hailo AI accelerator
- **Performance Optimized**: Multi-threaded architecture achieving 25-30 FPS on Raspberry Pi 5

## Technical Stack

**Languages & Frameworks:**
- Python 3.8+
- OpenCV (Computer Vision)
- YOLOv8 (Ultralytics)
- NumPy (Mathematical computations)

**Hardware Platforms:**
- Desktop/Laptop (Windows, Linux, macOS)
- Raspberry Pi 5 with camera module
- Optional: Hailo-8L AI accelerator, MPU6050 IMU sensor

## Project Structure

```
Aimbot-Bazooka/
├── Scripts/
│   ├── main_multithread.py          # Raspberry Pi optimized multithreaded version
│   ├── main_single_thread_pc.py     # PC/Laptop single-threaded version
│   ├── hailo_airsoft_test.py        # Hardware-accelerated version using HailoAI's own pipeline
│   ├── airsoft_test.py              # Field testing script
│   └── raspi_gyro_test.py           # IMU sensor integration test
├── Models/
│   └── yolov8n.pt                   # Pre-trained YOLO weights
├── Media/                           # Demo videos and images
└── requirements.txt
└── README.md
```

## Core Algorithms

### 1. Distance Estimation
Uses pinhole camera model for real-time distance calculation:
```
Distance = (Focal_Length × Real_Object_Width) / Detected_Object_Width
```

### 2. Ballistic Trajectory Simulation
Euler integration method accounting for:
- Gravitational acceleration
- Aerodynamic drag (velocity-dependent)
- Projectile mass and cross-sectional area

### 3. Predictive Targeting
Combines:
- Target velocity estimation from position history
- Flight time calculation from ballistic simulation
- Lead point prediction for moving targets

### 4. Camera Stabilization
ORB (Oriented FAST and Rotated BRIEF) feature detection with affine transformation to compensate for camera movement.

## Quick Start

```bash
# Clone repository
git clone https://github.com/ElVarsov/Aimbot-Bazooka.git
cd Aimbot-Bazooka

# Install dependencies
pip install -r requirements.txt

# Run on PC
python Scripts/main_single_thread_pc.py

# Run on Raspberry Pi 5
python Scripts/main_multithread.py
```

## Key Achievements

✅ **Multi-threaded Architecture**: Separate video processing and object detection threads for optimal performance  
✅ **Hardware Acceleration**: Support for Hailo AI accelerator achieving 2x speed improvement  
✅ **Real-Time Processing**: Maintains 30 FPS with concurrent ballistic calculations  
✅ **Cross-Platform**: Single codebase supporting PC, Linux, and embedded systems  
✅ **Modular Design**: Easy to extend with additional sensors (IMU)  

## Technical Highlights

**Computer Vision:**
- Custom YOLO detection pipeline with configurable confidence thresholds
- Grayscale optimization for 66% data reduction
- ORB feature matching for sub-pixel camera motion tracking

**Embedded Systems:**
- CPU affinity optimization for Raspberry Pi 5's quad-core architecture
- Queue-based frame management preventing memory overflow
- Priority scheduling for time-critical processing

**Physics Simulation:**
- Numerical integration with adaptive timestep
- Realistic drag coefficient modeling (G1 ballistic standard compatible)
- Sub-meter accuracy at ranges up to 50m

## Applications

- **Educational**: Demonstrates computer vision, physics simulation, and real-time processing
- **Research**: Platform for testing trajectory prediction algorithms
- **Hackathon**: Made as a project for the 2025 Define X SAKO Defense Hackathon in Riihimäki

## Future Development

- [ ] Integration with thermal imaging cameras (in cases where computer vision fails)
- [ ] Succesful integration of IMU 
- [ ] Integrating HailoAI more efficiently (Hailo driver issues)

---

**Developed for**: Define Hackathon X Sako  
**Repository**: [github.com/ElVarsov/Aimbot-Bazooka](https://github.com/ElVarsov/Aimbot-Bazooka)    
**Contact**: Available via GitHub Issues
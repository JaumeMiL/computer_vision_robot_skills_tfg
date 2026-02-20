# A Novel Computer Vision Framework to Enhance Robot Skills 
## TFG - Grau en Intel·ligència Artificial
### Author: Jaume Mora i Ladària
### Supervisors: Anaís Garrell and Isiah Zaplana
### Grade: 9.5/10
---

### Overview
This project presents the development of a robust teleoperation framework that enables the control of a TurtleBot3 Waffle Pi mobile robot using natural hand gestures. Implemented on ROS 2 Jazzy Jalisco, the system utilizes a distributed architecture where vision processing is offloaded to a remote workstation. By combining MediaPipe landmark extraction with the EurekaNet deep learning model, the framework provides a low-latency, controller-free interface suitable for non-expert users.

---

### 1. Framework Architecture and Components
The system is structured as a modular pipeline that transforms raw RGB video streams into precise robotic motion commands through three main stages.

#### Data Processing and Perception
* Hand Landmarks: Real-time extraction of 21 3D coordinates using MediaPipe Hands.

* Feature Encoding: Creation of a 12,348-dimensional DistTime vector representing temporal differences across 15 keyframes.

#### Decision and Control
* Classification: Inference via EurekaNet, a Multi-Layer Perceptron (MLP) trained on the IPN Hand dataset.

* Command Mapping: Translation of gesture IDs into TwistStamped messages for linear and angular velocity control.

#### Safety Layers
* LiDAR Monitoring: Frontal sector scanning (20 degree field) for automatic obstacle avoidance.

* Conservative Inference: Implementation of confidence thresholds and temporal voting (majority of 4/7 frames) to prevent unintended activations.

---

### 2. Phase I: Geometric Heuristics (Baseline Prototype)
The initial phase established a deterministic baseline using handcrafted geometric rules. This model served to validate the ROS 2 communication flow and basic motor responses.

| Feature            | Value / Status                         |
| :----------------- | :------------------------------------- |
| Core Logic         | Handcrafted geometric rules            |
| Gesture Diversity  | 4 Static + 1 Heuristic Dynamic         |
| Robustness         | High sensitivity to hand orientation   |
| Computational Cost | Minimal (CPU-only)                     |

---

### 3. Phase II: Deep Learning Model (EurekaNet)
To address the limitations of heuristics, the system was migrated to the EUREKA architecture. This model utilizes spatio-temporal patterns to ensure invariance to hand rotation and distance.

* Batch Approach: Simultaneous processing of 30, 60, and 90-frame windows to detect gestures of variable duration.
* Hardware Acceleration: Model trained for 100 epochs using MPS (Metal Performance Shaders) on Apple Silicon.

| Metric                        | Result                         |
| :---------------------------- | :----------------------------- |
| Training Accuracy             | 93.83%                         |
| Training Loss                 | 0.1774                         |
| Global Real-World Accuracy    | 91.17% (based on 600 trials)   |
| Inference Latency             | < 25 ms                        |

---

### 4. Conclusions and System Effectiveness
* Real-Time Performance: The distributed setup achieves an end-to-end latency of 200 to 300 ms over WiFi, which is well within the requirements for safe robot teleoperation.
* Interaction Reliability: The integration of a locking mechanism and LiDAR-based safety constraints ensures that the robot maintains a conservative behavior, effectively ignoring visual noise and preventing collisions.
* Scalability and Modernization: The migration to ROS 2 Jazzy Jalisco ensures compatibility with current robotics standards and provides a foundation for future multimodal integrations (voice and vision).
* Economic Viability: The project validates that high-precision control can be achieved using standard RGB hardware and open-source frameworks, significantly reducing the cost of assistive robotics.

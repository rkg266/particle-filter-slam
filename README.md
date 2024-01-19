# particle-filter-slam
Implemented simultaneous localization and mapping (SLAM), leveraging particle filter strategy for estimating the SLAM pose of a differential-drive robot.

## Data utilized:
1. Encoder: Linear velocity.
2. IMU: Linear acceleration and angular velocity.
3. LiDAR: Distances of obstacles in the environment.
4. Kinect: RGB and disparity images.
   
## Tasks:
1. Estimating the position of the robot over time using particle filter. <br>
**Results:** Deadreckon and estimated trajectories shown below <br>
![Deadreackon](/plots_images/drkn_traject_20.png) <br>
![Particle filter](/plots_images/Ptraject_20.png)

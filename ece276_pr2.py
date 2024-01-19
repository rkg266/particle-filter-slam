import os
import pickle
from PR2_functions import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bresenham import bresenham
from tqdm import tqdm

##### USER EDITABLE ####
path_data = r'/home/renukrishna/ece276a/Project2/data'
path_dataRGBD = r'/home/renukrishna/ece276a/Project2/dataRGBD'
disparity_dir = ['Disparity20', 'Disparity21']
rgb_dir = ['RGB20', 'RGB21']
ids_data = [20, 21]
#########
DDRdata_sets = readDDRData(path_data)

for i in ids_data:
    cur_set = DDRdata_sets[i]
    cur_set.cnvrtLidar2XY_Bodyfrm()  # Lidar ranges -> body frame coordinates
    cur_set.FilterYawRate_IMU(cutoff=10, fsamp=2000, order=5) # Low pass filter on Yaw data
    plt.figure()
    plt.plot(cur_set.imu_angular_velocity[2, :])
    plt.xlabel('time')
    plt.ylabel('Yaw rate')
    plt.title('Unfiltered yaw rate #' + str(i))
    plt.figure()
    plt.plot(cur_set.yaw_rate_filtered)
    plt.title('Filtered yaw rate #' + str(i))
    plt.xlabel('time')
    plt.ylabel('Yaw rate')
    cur_set.ComputeLinearVelocities() # Compute linear velocities from Encoder data
    
    # Log-odds
    prcnt_trust = 80
    log_odd = np.log(prcnt_trust/(100-prcnt_trust))
    thresh = 1e3
    lb = -thresh*log_odd # lower bound
    ub = -lb  # upper bound
##-------------------------------------------------------------------------------

    # DEAD RECKONING
    cur_set.RunDDrKinematicModel() # Run deadreckoning to compute noiseless trajectory
    traject = cur_set.NoiselessTrajectory
    cur_set.DeadRknOccupancyMap() # Construct occupancy grid map based on deadreckoning
    occu_map = cur_set.DdRknOcpncyMap
    np.savez('DedrknOccuMap_'+str(i)+ '.npz', occu_map=occu_map)

    plt.figure()
    plt.plot(traject[0, :], traject[1, :])
    plt.title('Deadreckon: Trajectory #' + str(i))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.figure()
    plt.plot(traject[2, :])
    plt.title('Deadreckon: Yaw (Theta) #' + str(i))
    plt.xlabel('time')
    plt.ylabel('Yaw angle (rad)')
    plt.show()
    plt.figure()
    with np.load('DedrknOccuMap_'+str(i)+ '.npz') as data:
        occu_map = data['occu_map']
    occu_map[occu_map < lb] = lb
    occu_map[occu_map > ub] = ub
    plt.imshow(occu_map.astype(np.int16), cmap='gray')
    plt.title('Deadreckon: Occupany Map #' + str(i))
    plt.show()
##-------------------------------------------------------------------------------

    # PARTICLE FILTER SLAM
    cur_set.Predict_Update_Particles(NumParticles=5, noise_sd=(0.05, 0.001)) ##### USER EDITABLE ####
    prtcl_est_map = cur_set.partilceEstMap
    prtcl_est_traj = cur_set.partEstTrajectory
    fnme = 'PrtclMapTraj_' + str(i) + '.npz'
    np.savez(fnme, prtcl_est_map=prtcl_est_map, prtcl_est_traj=prtcl_est_traj)

    with np.load(fnme) as data:
        prtcl_est_map = data['prtcl_est_map']
        prtcl_est_traj = data['prtcl_est_traj']
    prtcl_est_map[prtcl_est_map < lb] = lb
    prtcl_est_map[prtcl_est_map > ub] = ub
    prtcl_est_map_bin = np.zeros(prtcl_est_map.shape)
    prtcl_est_map_bin[prtcl_est_map>0] = 1
    plt.figure()
    plt.imshow(prtcl_est_map, cmap='gray')
    #plt.imshow(prtcl_est_map_bin, cmap='gray')
    plt.title('Particle estimated Occupany Map #' + str(i))
    plt.figure()
    plt.plot(prtcl_est_traj[0, :], prtcl_est_traj[1, :])
    plt.title('Particle estimated Trajectory #' + str(i))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
##-------------------------------------------------------------------------------

    # TEXTURE MAP
    cur_set.TextureMap = Texture_map(cur_set, path_dataRGBD, prtcl_est_traj)
    fnme1 = 'TextureMap_' + str(i) + '.npz'
    np.savez(fnme1, TextureMap=cur_set.TextureMap)
    plt.figure()
    with np.load(fnme1) as data:
        cur_set.TextureMap = data['TextureMap']
    plt.title('Texture Map #' + str(i))
    plt.imshow(cur_set.TextureMap)
    bs = 4 # Stop here to verify plots of one dataset
bh=9


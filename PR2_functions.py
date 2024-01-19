import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from bresenham import bresenham
from tqdm import tqdm
from scipy.signal import butter, lfilter
from pr2_utils import *
import cv2 as cv2

class ddr_dataset:
    index=0
    encoder_counts=[]
    encoder_stamps=[]
    encoder_delta_t = []
    left_velocity=[]
    right_velocity=[]
    linear_velocity=[]
    lidar_angle_min = []
    lidar_angle_max = []
    lidar_angle_increment = []
    lidar_range_min = []
    lidar_range_max = []
    lidar_ranges = []
    lidar_stamps = []	
    imu_angular_velocity = []
    imu_linear_acceleration = []
    imu_stamps = []
    disp_stamps = []
    rgb_stamps = []

    lidar_snsrX = []
    lidar_snsrY = []
    lidar_bodyX = []
    lidar_bodyY = []

    yaw_rate_filtered = []
    NoiselessTrajectory=[]
    DdRknOcpncyMap = []
    partilceEstOcpncyMap = []
    partEstTrajectory = []
    TextureMap = []

    map_res = 0.05 #meters
    map_xmin = -30  #meters
    map_ymin = -30
    map_xmax = 30
    map_ymax = 30 
######################################## IN CLASS FUNCTIONS ##########################################
    def cnvrtLidar2XY_Bodyfrm(self):
        #p_snsr_wrt_centr = np.array([0.13323, 0])
        p_snsr_wrt_centr = np.array([0, 0])
        npts = self.lidar_ranges.shape[0] # 1081
        nstmps = self.lidar_ranges.shape[1]
        ids = np.array(range(npts))
        inc = self.lidar_angle_increment * ids
        theta = self.lidar_angle_min + inc
        cos_theta = np.diag(np.cos(theta).flatten())
        sin_theta = np.diag(np.sin(theta).flatten())
        bool_min = np.where(self.lidar_ranges < self.lidar_range_min)
        bool_max = np.where(self.lidar_ranges > self.lidar_range_max)
        cur_ranges = self.lidar_ranges
        cur_ranges[bool_min[0], bool_min[1]] = 0
        cur_ranges[bool_max[0], bool_max[1]] = 0
        self.lidar_snsrX = np.matmul(cos_theta, cur_ranges)
        self.lidar_snsrY = np.matmul(sin_theta, cur_ranges)
        self.lidar_bodyX = self.lidar_snsrX + p_snsr_wrt_centr[0]
        self.lidar_bodyY = self.lidar_snsrY + p_snsr_wrt_centr[1]
    
    def ComputeLinearVelocities(self):
        dX = 0.0022 #mtr
        fr=0
        fl=1
        rr=2
        rl=3
        encdr_cnts = self.encoder_counts
        encdr_t = self.encoder_stamps
        encdr_cnts = encdr_cnts[:, 1:] # leaving 1st count. All counts are reset every measure
        encdr_delta_t = np.diff(encdr_t, n=1)
        dist_L = ((encdr_cnts[fl, :] + encdr_cnts[rl, :])/2)*dX # mtr
        dist_R = ((encdr_cnts[fr, :] + encdr_cnts[rr, :])/2)*dX # mtr
        self.left_velocity = dist_L/encdr_delta_t
        self.right_velocity = dist_R/encdr_delta_t
        self.linear_velocity = (self.left_velocity + self.right_velocity)/2
        self.encoder_delta_t = encdr_delta_t
    
    def FilterYawRate_IMU(self, cutoff, fsamp, order):
        yaw_rate = self.imu_angular_velocity[2, :] # Wz in body frame rad/s
        yaw_rate_filt = low_pass_filter(yaw_rate, cutoff, fsamp, order)
        self.yaw_rate_filtered = yaw_rate_filt

    def RunDDrKinematicModel(self):
        lin_V = self.linear_velocity 
        V_t = self.encoder_stamps[1:] # leaving the 1st timestamp 
        tau = self.encoder_delta_t
        W_t = self.imu_stamps # Angular velocity timestamps
        yaw_rate = self.yaw_rate_filtered # Wz in body frame
        V_nrby_W = np.searchsorted(W_t, V_t)
        V_nrby_W = V_nrby_W - 1
        V_nrby_W[V_nrby_W < 0] = 0
        self.NoiselessTrajectory = RunMotionModel(V_nrby_W, tau, lin_V, yaw_rate, noise_sd=(0,0))

    def Predict_Update_Particles(self, NumParticles, noise_sd):
        lin_V = self.linear_velocity # Linear velocities from Encoder
        V_t = self.encoder_stamps[1:] # leaving the 1st timestamp 
        tau = self.encoder_delta_t
        W_t = self.imu_stamps # Angular velocity timestamps
        ldr_bodyX = self.lidar_bodyX
        ldr_bodyY = self.lidar_bodyY
        ldr_t = self.lidar_stamps # LiDAR time stamps
        yaw_rate = self.yaw_rate_filtered # Wz in body frame

        V_nrby_W = np.searchsorted(W_t, V_t)
        V_nrby_W = V_nrby_W - 1
        V_nrby_W[V_nrby_W < 0] = 0
        yaw_nrby = yaw_rate[V_nrby_W] # Yaw rates near to velocity encoder timestamps
        V_nrby_ldr = np.searchsorted(ldr_t, V_t)
        V_nrby_ldr = V_nrby_ldr - 1
        V_nrby_ldr[V_nrby_ldr < 0] = 0 # Lidar scans near to velocity encoder timestamps
        ldr_bodyX_nrby = ldr_bodyX[:, V_nrby_ldr]
        ldr_bodyY_nrby = ldr_bodyY[:, V_nrby_ldr]

        N = NumParticles
        # init MAP
        MAP = {}
        MAP['res']   = self.map_res #meters
        MAP['xmin']  = self.map_xmin #meters
        MAP['ymin']  = self.map_ymin
        MAP['xmax']  = self.map_xmax
        MAP['ymax']  = self.map_ymax 
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) 
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'])) # log-odd MAP
        MAP['binmap'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8. Binary map

        prcnt_trust = 80
        log_odd = np.log(prcnt_trust/(100-prcnt_trust)) # must be positive
        thresh = 1e3
        lb = -thresh*log_odd # must be negative
        ub = -lb
        p_N = np.zeros((N, 3)) # Particle poses initialized
        wt_N = np.ones((N,))/N # particle weights

        #  Map using 1st lidar scan
        xs0 = ldr_bodyX_nrby[:, 0] # Lidar body frame coordinates in mtrs
        ys0 = ldr_bodyY_nrby[:, 0]
        theta_N = p_N[0, 2] # Assuming 1st particle as max weight
        cos_theta_N = np.cos(theta_N)
        sin_theta_N = np.sin(theta_N)
        cos_bodyX = xs0 * cos_theta_N
        sin_bodyX = xs0 * sin_theta_N
        cos_bodyY = ys0 * cos_theta_N
        sin_bodyY = ys0 * sin_theta_N 
        xw0 = (cos_bodyX - sin_bodyY) + p_N[0, 0] # world coordinates
        yw0 = (sin_bodyX + cos_bodyY) + p_N[0, 1]
        xw_pix = np.ceil((xw0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 # Converting world mtrs to pixels
        yw_pix = np.ceil((yw0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        # Build occupancy map
        for i in range(xw_pix.shape[0]):
            robX = p_N[0, 0]
            robY = p_N[0, 1]
            robX_pix = np.ceil((robX - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 # Converting world mtrs to pixels
            robY_pix = np.ceil((robY - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
            cur_ldr_wx_pix = xw_pix[i]
            cur_ldr_wy_pix = yw_pix[i]
            if cur_ldr_wx_pix == robX_pix and cur_ldr_wy_pix == robY_pix: # reject out of range
                continue
            path_xy = bresenham2D(robX_pix, robY_pix, cur_ldr_wx_pix, cur_ldr_wy_pix).astype(np.int16)
            MAP['map'][path_xy[0, 0:-1], path_xy[1, 0:-1]] = MAP['map'][path_xy[0, 0:-1], path_xy[1, 0:-1]] - log_odd
            MAP['map'][path_xy[0, -1], path_xy[1, -1]] = MAP['map'][path_xy[0, -1], path_xy[1, -1]] + log_odd
        #plt.imshow(MAP['map'], cmap='gray')
        MAP['binmap'][MAP['map'] > 0] = 1
        MAP['binmap'][MAP['map'] < 0] = 0

        x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-mtrs of each pixel of the map
        y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-mtrs of each pixel of the map
        mapres = MAP['res']
        x_range = np.arange(-0.2,0.2+mapres,mapres) # 9x9 window
        y_range = np.arange(-0.2,0.2+mapres,mapres)
        update_steps = 5
        est_traj = np.zeros((3, len(tau)))
        est_traj[:, 0] = p_N[0, :] 

        # Run through each time sample
        for itr in tqdm(range(len(tau)-1)):
            # Predict next particle poses
            cur_tau = tau[itr]
            cur_V = lin_V[itr]
            cur_yaw = yaw_nrby[itr]
            p_N = Run_Predict_Particles(p_N, cur_tau, cur_V, cur_yaw, noise_sd)
            # Update trajectory
            max_n = np.argmax(wt_N)
            est_traj[:, itr+1] = p_N[max_n, :]

            # Current lidar scan 
            xs0 = ldr_bodyX_nrby[:, itr+1] # body frame coordinates in mtrs
            ys0 = ldr_bodyY_nrby[:, itr+1]

            if (itr % update_steps) == 0:
                store_ldr_wrldXY_N = np.zeros((N, 2, xs0.shape[0]))
                for n in range(p_N.shape[0]): # Each particle
                    theta = p_N[n, 2] 
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    cos_bodyX = xs0 * cos_theta
                    sin_bodyX = xs0 * sin_theta
                    cos_bodyY = ys0 * cos_theta
                    sin_bodyY = ys0 * sin_theta 
                    xw0 = (cos_bodyX - sin_bodyY) + p_N[n, 0] # world coordinates of lidar points w.r.t one particle
                    yw0 = (sin_bodyX + cos_bodyY) + p_N[n, 1]
                    cur_lidar_XY = np.stack((xw0,yw0))
                    store_ldr_wrldXY_N[n, :, :] = cur_lidar_XY

                    c = mapCorrelation(MAP['binmap'], x_im, y_im, cur_lidar_XY, x_range, y_range)
                    cmax_id = np.argmax(c)
                    c_id0 = int(cmax_id/9)
                    c_id1 = int(cmax_id%9)
                    cmax = c[c_id0, c_id1]
                    wt_N[n] = wt_N[n] * cmax # Particle weight update
                    # p_N[n, 0] = p_N[n, 0] + x_range[c_id0] # Improvement to particle location
                    # p_N[n, 1] = p_N[n, 1] + y_range[c_id1]
                wt_N = wt_N/np.sum(wt_N) # Normalizing weights

                # Update map w.r.t max weight particle
                max_n = np.argmax(wt_N)
                ldr_wrldX =  store_ldr_wrldXY_N[max_n, 0, :]
                ldr_wrldY =  store_ldr_wrldXY_N[max_n, 1, :]

                xw_pix = np.ceil((ldr_wrldX - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 # Converting world mtrs to pixels
                yw_pix = np.ceil((ldr_wrldY - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
                # Update occupancy map
                for i in range(xw_pix.shape[0]):
                    robX = p_N[max_n, 0] # Position of max weight particle
                    robY = p_N[max_n, 1]
                    robX_pix = np.ceil((robX - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 # Converting world mtrs to pixels
                    robY_pix = np.ceil((robY - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
                    cur_ldr_wx_pix = xw_pix[i]
                    cur_ldr_wy_pix = yw_pix[i]
                    if cur_ldr_wx_pix == robX_pix and cur_ldr_wy_pix == robY_pix: # reject 
                        continue
                    path_xy = bresenham2D(robX_pix, robY_pix, cur_ldr_wx_pix, cur_ldr_wy_pix).astype(np.int16)
                    MAP['map'][path_xy[0, 0:-1], path_xy[1, 0:-1]] = MAP['map'][path_xy[0, 0:-1], path_xy[1, 0:-1]] - log_odd
                    MAP['map'][path_xy[0, -1], path_xy[1, -1]] = MAP['map'][path_xy[0, -1], path_xy[1, -1]] + log_odd
                #plt.imshow(MAP['map'], cmap='gray')   
                MAP['binmap'][MAP['map'] > 0] = 1
                MAP['binmap'][MAP['map'] < 0] = 0

                # Check for resampling
                rethresh = int(NumParticles/2)
                N = len(wt_N)
                Neff = 1/np.square(np.linalg.norm(wt_N))
                if Neff < rethresh:
                    prt = np.random.choice(N, N, p=wt_N, replace=True)
                    p_N = p_N[prt, :] # Update particle poses
                    wt_N = np.ones((N,))/N # Initialize particle weights
        MAP['map'][MAP['map'] < lb] = lb 
        MAP['map'][MAP['map'] > ub] = ub 
        self.partilceEstMap = MAP['map']
        self.partEstTrajectory = est_traj

    def DeadRknOccupancyMap(self):
        MAP = {}
        MAP['res']   = self.map_res #meters
        MAP['xmin']  = self.map_xmin #meters
        MAP['ymin']  = self.map_ymin
        MAP['xmax']  = self.map_xmax
        MAP['ymax']  = self.map_ymax
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'])) 

        ldr_stamps = self.lidar_stamps 
        ldr_bodyX = self.lidar_bodyX
        ldr_bodyY = self.lidar_bodyY

        traject = self.NoiselessTrajectory
        traj_stamps = self.encoder_stamps 
        traj_nrby_ldr = np.searchsorted(ldr_stamps, traj_stamps)
        traj_nrby_ldr = traj_nrby_ldr - 1
        traj_nrby_ldr[traj_nrby_ldr < 0] = 0

        # Robot position pixel coordinates
        xy_traject = traject[0:2, :]
        rob_x_px = np.ceil((xy_traject[0, :] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        rob_y_px = np.ceil((xy_traject[1, :] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

        # Lidar points world pixel coordinates
        ldr_bodyX_nrby = ldr_bodyX[:, traj_nrby_ldr]
        ldr_bodyY_nrby = ldr_bodyY[:, traj_nrby_ldr]
        theta_rob = traject[2, :]
        cos_theta_mat = np.diag(np.cos(theta_rob)) 
        sin_theta_mat = np.diag(np.sin(theta_rob)) 
        cos_ldr_Bx = np.matmul(ldr_bodyX_nrby, cos_theta_mat) # cos(theta)*x_b
        cos_ldr_By = np.matmul(ldr_bodyY_nrby, cos_theta_mat) # cos(theta)*y_b
        sin_ldr_Bx = np.matmul(ldr_bodyX_nrby, sin_theta_mat) # sin(theta)*x_b
        sin_ldr_By = np.matmul(ldr_bodyY_nrby, sin_theta_mat) # sin(theta)*y_b
        ldr_worldX_nrby = (cos_ldr_Bx - sin_ldr_By) + xy_traject[0, :] # world coordinates
        ldr_worldY_nrby = (sin_ldr_Bx + cos_ldr_By) + xy_traject[1, :]

        ldr_x_px = np.ceil((ldr_worldX_nrby - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 # Ldr World frame pixel positions
        mxx = np.max(ldr_x_px)
        mnx = np.min(ldr_x_px)
        ldr_y_px = np.ceil((ldr_worldY_nrby - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        mxy = np.max(ldr_y_px)
        mny = np.min(ldr_y_px)

        prcnt_scs = 80
        log_odd = np.log(prcnt_scs/(100-prcnt_scs))
        thresh = 1e3
        lb = -thresh*log_odd
        ub = -lb
        for i in tqdm(range(len(rob_x_px))): # x, y are same length
            cur_lidarX = ldr_x_px[:, i]
            cur_lidarY = ldr_y_px[:, i]
            for j in range(len(cur_lidarX)):
                if cur_lidarX[j] == rob_x_px[i] and cur_lidarY[j] == rob_y_px[i]:
                    continue
                seg_list = bresenham2D(rob_x_px[i], rob_y_px[i], cur_lidarX[j], cur_lidarY[j]).astype(np.int16)
                #seg_list = np.transpose(np.array(list(bresenham(rob_x_px[i], rob_y_px[i], cur_lidarX[j], cur_lidarY[j]))))
                MAP['map'][seg_list[0, 0:-1], seg_list[1, 0:-1]] = MAP['map'][seg_list[0, 0:-1], seg_list[1, 0:-1]] - log_odd
                MAP['map'][seg_list[0, -1], seg_list[1, -1]] = MAP['map'][seg_list[0, -1], seg_list[1, -1]] + log_odd
        # MAP['map'][MAP['map'] < lb] = lb
        # MAP['map'][MAP['map'] > ub] = ub    
        self.DdRknOcpncyMap = MAP['map']

######################################## GENERAL FUNCTIONS ##########################################
def Texture_map(cur_set, rgbd_path, slam_traject):
    data_id = cur_set.index - 20
    dirs_list_all = sorted(os.listdir(rgbd_path))
    disparity_dir = [x for x in dirs_list_all if x.startswith('Disparity')]
    rgb_dir = [x for x in dirs_list_all if x.startswith('RGB')]
    
    disp_path = os.path.join(rgbd_path, disparity_dir[data_id])
    rgb_path = os.path.join(rgbd_path, rgb_dir[data_id])
    disp_list_all = os.listdir(disp_path)
    rgb_list_all = os.listdir(rgb_path)
    disp_list = [x for x in disp_list_all if x.endswith('.png')]
    rgb_list = [x for x in rgb_list_all if x.endswith('.png')]
    disp_list = sorted(disp_list, key=lambda x: int(x[x.find('_')+1:x.find('.png')]))
    rgb_list = sorted(rgb_list, key=lambda x: int(x[x.find('_')+1:x.find('.png')]))
    disp_t = cur_set.disp_stamps # required disparity stamps (anchor)
    rgb_t = cur_set.rgb_stamps
    slam_t = cur_set.encoder_stamps[0:-1]
    #slam_poses = cur_set.NoiselessTrajectory
    slam_poses = slam_traject

    dispt_in_rgbt = np.searchsorted(rgb_t, disp_t) # Every disp image needs corresponding rgb image
    dispt_in_rgbt = dispt_in_rgbt - 1
    dispt_in_rgbt[dispt_in_rgbt < 0] = 0
    rgb_t_dsp = dispt_in_rgbt # required RGB stamps
    dispt_in_slamt = np.searchsorted(slam_t, disp_t) # Every disp image needs corresponding slam pose
    dispt_in_slamt = dispt_in_slamt - 1
    dispt_in_slamt[dispt_in_slamt < 0] = 0
    slam_t_dsp = dispt_in_slamt # required slam stamps
    
    if cur_set.DdRknOcpncyMap == []:
        with np.load(occupancy_map_file) as data:
            occu_map = data['prtcl_est_map']
    else:
        occu_map = cur_set.DdRknOcpncyMap 

    FLOORMAP = {}
    FLOORMAP['res']   = cur_set.map_res 
    FLOORMAP['xmin']  = -50
    FLOORMAP['ymin']  = -50
    FLOORMAP['xmax']  = 50
    FLOORMAP['ymax']  = 50
    # FLOORMAP['xmin']  = cur_set.map_xmin 
    # FLOORMAP['ymin']  = cur_set.map_ymin
    # FLOORMAP['xmax']  = cur_set.map_xmax
    # FLOORMAP['ymax']  = cur_set.map_ymax
    FLOORMAP['sizex']  = int(np.ceil((FLOORMAP['xmax'] - FLOORMAP['xmin']) / FLOORMAP['res'] + 1)) #cells
    FLOORMAP['sizey']  = int(np.ceil((FLOORMAP['ymax'] - FLOORMAP['ymin']) / FLOORMAP['res'] + 1))
    FLOORMAP['rgbmap'] = np.zeros((FLOORMAP['sizex'],FLOORMAP['sizey'], 3))
    
    K = np.array([[585.05108211, 0, 242.94140713], [0, 585.05108211, 315.83800193], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    R_opt2reg = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    cam_orient = np.array([0, 0.36, 0.021]) # Roll, pitch, yaw
    cam_xyz_wrt_cntr = np.array([[0.18], [0.005], [0.36]]) #mtr
    z_ldr_wrt_cam = (514.35-380.01)/1000 # z-coordinate of lidar wrt cam in mtr

    for dp in tqdm(range(len(disp_t))):
        cur_disp_file = disp_list[dp]
        cur_rgb_file = rgb_list[rgb_t_dsp[dp]]
        # cur_disp = mpimg.imread(os.path.join(disp_path, cur_disp_file)).astype(np.float32)
        # cur_rgb = mpimg.imread(os.path.join(rgb_path, cur_rgb_file))
        # cur_disp = plt.imread(os.path.join(disp_path, cur_disp_file))
        # cur_rgb = plt.imread(os.path.join(rgb_path, cur_rgb_file))
        cur_disp = cv2.imread(os.path.join(disp_path, cur_disp_file), cv2.IMREAD_UNCHANGED)
        cur_disp = cur_disp.astype(np.float32)
        cur_rgb = cv2.imread(os.path.join(rgb_path, cur_rgb_file), cv2.IMREAD_UNCHANGED)
        cur_rgb = cur_rgb.astype(np.float32)
        cur_rgb = cur_rgb[:, :, [2, 1, 0]] # BGR -> RGB
        cur_pose = slam_poses[:, slam_t_dsp[dp]]
        cur_pose = cur_pose.reshape((3, 1))
        # plt.figure
        # plt.imshow(cur_disp)
        # plt.figure
        # plt.imshow(cur_rgb)

        dd = -0.00304*cur_disp + 3.31
        depth = 1.03/dd
        row_im = cur_disp.shape[0]
        col_im = cur_disp.shape[1]
        jt = np.array(range(col_im))
        j_ind = np.broadcast_to(jt, (row_im, col_im))
        it = np.array(range(row_im)).reshape((row_im, 1))
        i_ind = np.broadcast_to(it, (row_im, col_im))
        rgbi = np.round((526.37 * i_ind + (-4.5 * 1750.46)*dd + 19276.0) / 585.051).astype(np.int16) # Indices in RGB image
        rgbj = np.round((526.37 * j_ind + 16662) / 585.051).astype(np.int16)
        valid_rgb = np.logical_and(np.logical_and((rgbi >= 0), (rgbi < row_im)), np.logical_and((rgbj >= 0), (rgbj < col_im)))

        # Get 3D coordinates of disparity image
        fx = 585.05108211
        fy = 585.05108211
        cx = 315.83800193
        cy = 242.94140713
        d_x = (i_ind-cx) / fx * depth
        d_y = (j_ind-cy) / fy * depth
        d_z = depth

        D_xyz_0 = np.zeros((3, depth.shape[0], depth.shape[1]))
        D_xyz_0[0, :, :] = i_ind * depth
        D_xyz_0[1, :, :] = j_ind * depth
        D_xyz_0[2, :, :] = depth
        D_xyz_0_rs = D_xyz_0.reshape((3, D_xyz_0.shape[1]*D_xyz_0.shape[2]))
        D_xyz_opt_rs = np.matmul(K_inv, D_xyz_0_rs) # optical frame
        D_xyz_reg_rs = np.matmul(R_opt2reg, D_xyz_opt_rs) # regular frame

        rgbi_dpth = rgbi*depth
        rgbj_dpth = rgbj*depth
        rgb_xyz_0 = np.zeros((3, depth.shape[0], depth.shape[1]))
        rgb_xyz_0[0, :, :] = rgbi_dpth
        rgb_xyz_0[1, :, :] = rgbj_dpth
        rgb_xyz_0[2, :, :] = depth
        rgb_xyz_0_rs = rgb_xyz_0.reshape((3, rgb_xyz_0.shape[1]*rgb_xyz_0.shape[2]))
        rgb_xyz_opt_rs = np.matmul(K_inv, rgb_xyz_0_rs) # optical frame
        rgb_xyz_reg_rs = np.matmul(R_opt2reg, rgb_xyz_opt_rs) # regular frame

        # Regular camera frame to Body frame 
        cos_roll = np.cos(cam_orient[0])
        sin_roll = np.sin(cam_orient[0])
        cos_pitch = np.cos(cam_orient[1])
        sin_pitch = np.sin(cam_orient[1])
        cos_yaw = np.cos(cam_orient[2])
        sin_yaw = np.sin(cam_orient[2])
        Rx_roll = np.array([[1, 0, 0], [0, cos_roll, -sin_roll], [0, sin_roll, cos_roll]])
        Ry_pitch = np.array([[cos_pitch, 0, sin_pitch], [0, 1, 0], [-sin_pitch, 0, cos_pitch]])
        Rz_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
        R_tp = np.matmul(Ry_pitch, Rx_roll)
        R_cam2body = np.matmul(Rz_yaw, R_tp)
        rgb_xyz_body_rs = np.matmul(R_cam2body, rgb_xyz_reg_rs) + cam_xyz_wrt_cntr # body frame

        # Body to World frame
        cur_rob_xyz = np.zeros((3, 1))
        cur_rob_xyz[0:2] = cur_pose[0:2] # z=0
        cur_rob_orient = np.array([0, 0, cur_pose[2, 0]]) # roll, pitch, yaw
        cos_roll = np.cos(cur_rob_orient[0])
        sin_roll = np.sin(cur_rob_orient[0])
        cos_pitch = np.cos(cur_rob_orient[1])
        sin_pitch = np.sin(cur_rob_orient[1])
        cos_yaw = np.cos(cur_rob_orient[2])
        sin_yaw = np.sin(cur_rob_orient[2])
        Rx_roll = np.array([[1, 0, 0], [0, cos_roll, -sin_roll], [0, sin_roll, cos_roll]])
        Ry_pitch = np.array([[cos_pitch, 0, sin_pitch], [0, 1, 0], [-sin_pitch, 0, cos_pitch]])
        Rz_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
        R_tp = np.matmul(Ry_pitch, Rx_roll)
        R_body2world = np.matmul(Rz_yaw, R_tp)
        rgb_xyz_wrld_rs = np.matmul(R_body2world, rgb_xyz_body_rs) + cur_rob_xyz # World frame w.r.t robot center

        rgb_z_wrld_rs = rgb_xyz_wrld_rs[2, :] # z-coordinates
        floor_z = np.array(np.where((rgb_z_wrld_rs > -0.1) & (rgb_z_wrld_rs < 0.1))) # Below center level are assumed to be floor
        rgb_xy_wrld_floor = np.squeeze(rgb_xyz_wrld_rs[0:2, floor_z]).reshape((2, floor_z.shape[1])) # X,Y coordinates of floor RGB pixels

        rgb_floorx_pxl = np.ceil((rgb_xy_wrld_floor[0, :] - FLOORMAP['xmin']) / FLOORMAP['res'] ).astype(np.int16)-1
        rgb_floory_pxl = np.ceil((rgb_xy_wrld_floor[1, :] - FLOORMAP['ymin']) / FLOORMAP['res'] ).astype(np.int16)-1
        pi = np.floor(floor_z/col_im).astype(np.int16)
        pj = np.remainder(floor_z, col_im).astype(np.int16)
        floor_rgbi = rgbi[pi, pj] # Floor coordinates wrt RGB image
        floor_rgbj = rgbj[pi, pj]
        valid_floor_rgb = np.logical_and(np.logical_and((floor_rgbi >= 0), (floor_rgbi < row_im)),\
                                          np.logical_and((floor_rgbj >= 0), (floor_rgbj < col_im)))
        
        floor_rgbi_final = floor_rgbi[valid_floor_rgb] # floor related indices in RGB image
        floor_rgbj_final = floor_rgbj[valid_floor_rgb]
        rgb_floorx_pxl = rgb_floorx_pxl.reshape((1, len(rgb_floorx_pxl)))
        rgb_floory_pxl = rgb_floory_pxl.reshape((1, len(rgb_floory_pxl)))
        rgb_floorx_pxl = rgb_floorx_pxl[valid_floor_rgb] # pixel coordinates in floor map
        rgb_floory_pxl = rgb_floory_pxl[valid_floor_rgb]

        FLOORMAP["rgbmap"][rgb_floorx_pxl, rgb_floory_pxl, :] = cur_rgb[floor_rgbi_final, floor_rgbj_final, :]      
        # if dp == 500:
        #      plt.imshow(FLOORMAP["rgbmap"].astype(np.int16))

    #plt.imshow(FLOORMAP["rgbmap"].astype(np.int16))
    return FLOORMAP["rgbmap"].astype(np.int16)

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

def Run_Predict_Particles(p_N, cur_tau, cur_V, cur_yaw, noise_sd):
    N = p_N.shape[0] # number of particles
    v_sd = noise_sd[0]
    w_sd = noise_sd[1]
    cur_theta_N = p_N[:, 2] 
    cos_theta_N = np.cos(cur_theta_N)
    sin_theta_N = np.sin(cur_theta_N)
    noise_v_N = np.random.normal(0, v_sd, (N,)) # Noise for linear velocity
    noise_w_N = np.random.normal(0, w_sd, (N,)) # Noise angular velocity
    noisy_V_N = cur_V + noise_v_N # Noisy velocity
    noisy_w_N = cur_yaw + noise_w_N # Noisy yaw rate
    dx = cur_tau * noisy_V_N * cos_theta_N
    dy = cur_tau * noisy_V_N * sin_theta_N
    dw = cur_tau * noisy_w_N 
    next_p_N = np.zeros((N, 3))
    next_p_N[:, 0] = p_N[:, 0] + dx
    next_p_N[:, 1] = p_N[:, 1] + dy
    next_p_N[:, 2] = p_N[:, 2] + dw
    return next_p_N
  
def sigmoid(x):
    y = 1.0/(1.0 + np.exp(-x))
    return y

def low_pass_filter(data, cutoff, fsamp, order):
    nyq = 0.5 * fsamp
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def RunMotionModel(V_nrby_W, tau, lin_V, yaw_rate, noise_sd):
    v_sd = noise_sd[0]
    w_sd = noise_sd[1]
    noise_v = np.random.normal(0, v_sd, (1, len(tau))) # Noise for linear velocities
    noise_w = np.random.normal(0, w_sd, (1, len(tau))) # Noise angular velocity
    p = np.array([0., 0., 0.]) # x, y, theta - world frame
    traject = np.zeros((3, len(tau)+1))
    traject[:, 0] = p
    for i in range(len(tau)):
        theta = p[2] 
        noisy_V = lin_V[i] + noise_v[0, i]
        noisy_w = yaw_rate[V_nrby_W[i]] + noise_w[0, i]
        tp = np.array([noisy_V*np.cos(theta), noisy_V*np.sin(theta), noisy_w])
        p = p + tau[i] * tp
        traject[:, i+1] = p
    return traject

def readDDRData(datapath):
    files_list_all = sorted(os.listdir(datapath))
    files_list_npz = [x for x in files_list_all if x.endswith('.npz')]
    files_encoder = [x for x in files_list_npz if x.find('Encoders') != -1]
    files_hokuyo = [x for x in files_list_npz if x.find('Hokuyo') != -1]
    files_imu = [x for x in files_list_npz if x.find('Imu') != -1]
    files_kinect = [x for x in files_list_npz if x.find('Kinect') != -1]
    
    data_dict = dict()
    for fil in files_encoder:
        with np.load(os.path.join(datapath, fil)) as data:
            t1 = fil.find('.npz')
            id = int(fil[len('Encoders'):t1])
            if id not in data_dict.keys():
                tpset = ddr_dataset()
                tpset.index = id
            else:
                tpset = data_dict[id]
            tpset.encoder_counts = data["counts"] # 4 x n encoder counts
            tpset.encoder_stamps = data["time_stamps"] # encoder time stamps
            if id not in data_dict.keys():
                data_dict[id] = tpset

    for fil in files_hokuyo:
        with np.load(os.path.join(datapath, fil)) as data:
            t1 = fil.find('.npz')
            id = int(fil[len('Hokuyo'):t1])
            if id not in data_dict.keys():
                tpset = ddr_dataset()
                tpset.index = id
            else:
                tpset = data_dict[id]
            tpset.lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
            tpset.lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
            tpset.lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
            tpset.lidar_range_min = data["range_min"] # minimum range value [m]
            tpset.lidar_range_max = data["range_max"] # maximum range value [m]
            tpset.lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
            tpset.lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
            if id not in data_dict.keys():
                data_dict[id] = tpset

    for fil in files_imu:        
        with np.load(os.path.join(datapath, fil)) as data:
            t1 = fil.find('.npz')
            id = int(fil[len('Imu'):t1])
            if id not in data_dict.keys():
                tpset = ddr_dataset()
                tpset.index = id
            else:
                tpset = data_dict[id]
            tpset.imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
            tpset.imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
            tpset.imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
            if id not in data_dict.keys():
                data_dict[id] = tpset
            
    for fil in files_kinect:    
        with np.load(os.path.join(datapath, fil)) as data:
            t1 = fil.find('.npz')
            id = int(fil[len('Kinect'):t1])
            if id not in data_dict.keys():
                tpset = ddr_dataset()
                tpset.index = id
            else:
                tpset = data_dict[id]
            tpset.disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
            tpset.rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
            if id not in data_dict.keys():
                data_dict[id] = tpset
    return data_dict
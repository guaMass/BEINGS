import numpy as np
import torch
from utils import angle_mod
import os
from tqdm import tqdm
import PIL
import matplotlib.pyplot as plt
import math

currentdir = os.path.dirname(os.path.abspath(__file__))
beings_dir = os.path.dirname(currentdir)
gaussian_dir = os.path.join(beings_dir, "3DGS_PoseRender")
patchnet_dir = os.path.join(beings_dir, "patchnetvlad")
os.sys.path.insert(0, gaussian_dir)
os.sys.path.insert(0, patchnet_dir)
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer

from utils import rotation_matrix_y

import torchvision.transforms as transforms

from vlad_loss import VLAD_SIM
vlad_s = VLAD_SIM("patchnetvlad/patchnetvlad/configs/speed.ini")
vlad_p = VLAD_SIM("patchnetvlad/patchnetvlad/configs/performance.ini")

import rospy
import cv2
from sensor_msgs.msg import CameraInfo, Image 
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
import time

RUN_SIM = True
OFFLINE_MODE = False
# model_path = "scence/03/larger.ply" # Path to the ply file model
model_path = "scence/05/splat_k.ply" # Path to the ply file model
task_name = "image1"
target = PIL.Image.open(f"target/scence05/{task_name}.png")
control_duration = 2.0

class ROS_Interface():
    def __init__(self):
        rospy.init_node('beings')
        self.bridge = CvBridge()  
        self.color_image = None
        
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
    def acquire_color_image(self):  
        color_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        self.color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        return self.color_image
    
    def acquire_odom(self):  
        odom_msg = rospy.wait_for_message("/ranger_base_node/odom", Odometry)

        x = odom_msg.pose.pose.position.x
        y = -odom_msg.pose.pose.position.y
        orientation_q = odom_msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        return [x, y, 0 ,yaw]
    
    def acquire_哦菩提擦亮_pose(self):  
        pose_msg = rospy.wait_for_message("/vrpn_client_node/car/pose", PoseStamped)

        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        orientation_q = pose_msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        return [x, y, 0 ,yaw]

if not RUN_SIM or OFFLINE_MODE:
    ros_interface = ROS_Interface()

camera = Camera()
# camera_info = {'width': 1920,
#                 'height': 1440,
#                 'position': [0, -0.5, 0],
#                 'rotation': [[1,0,0],[0,1,0],[0,0,1]],
#                 'fy': 1371.7027360999618,
#                 'fx': 1371.7027360999618}
camera_info = {'width': 2560,
                'height': 1920,
                'position': [0, 0, 0],
                'rotation': [[1,0,0],[0,1,0],[0,0,1]],
                'fy': 2420.915039062,
                'fx': 2420.099853516}
camera.load(camera_info)
gaussian_model = GaussianModel().load(model_path)
renderer = Renderer(gaussian_model, camera, logging=False)

def input_transform(resize=(480, 640)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

it = input_transform((480, 640))
tf_target = it(target).unsqueeze(0).cuda()

# if temp/task_name not exist, create it
if not os.path.exists(f'temp/{task_name}'):
    os.makedirs(f'temp/{task_name}')

# def transform_state(state):
#     # new_state has same shape with state
#     new_state = state.copy()
#     # rotate the state 180 degree along y-axis
#     new_state[:,:,:3] = -state[:,:,:3]
#     # switch new_state[:,:,1] and new_state[:,:,2]
#     new_state[:,:,1], new_state[:,:,2] = new_state[:,:,2].copy(), new_state[:,:,1].copy()
#     new_state[:,:,3] = -(state[:,:,3]+np.pi/2)
#     return new_state


def transform_matrices_batch(input_tensor):
    # 假设输入为 (N, K, 4) 的张量
    x, y, z, w = input_tensor[..., 0], input_tensor[..., 1], input_tensor[..., 2], input_tensor[..., 3]
    
    # 计算 cos(w) 和 sin(w)
    cos_w = torch.cos(w)
    sin_w = torch.sin(w)
    
    # 初始化 (N, K, 4, 4) 的输出张量
    N, K = input_tensor.shape[:2]
    matrices = torch.zeros((N, K, 4, 4), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # 填充旋转和平移矩阵
    matrices[..., 0, 0] = cos_w
    matrices[..., 0, 1] = -sin_w
    matrices[..., 1, 0] = sin_w
    matrices[..., 1, 1] = cos_w
    matrices[..., 2, 2] = 1
    matrices[..., 3, 3] = 1
    
    # 设置平移部分
    matrices[..., 0, 3] = x
    matrices[..., 1, 3] = y
    matrices[..., 2, 3] = z
    
    return matrices

def apply_transformation(input_tensor, T):
    # 假设 input_tensor 为 (N, K, 4, 4)，T 为 (4, 4)

    T = torch.from_numpy(T)
    
    # 将 T 扩展成 (N, K, 4, 4) 的形状，便于批量矩阵乘法
    T_expanded = T.expand(input_tensor.shape[0], input_tensor.shape[1], 4, 4)
    
    # 执行批量矩阵乘法
    result = torch.matmul(input_tensor, T_expanded)
    
    return result

def extract_position_and_angle(transformed_tensor):
    # 假设 transformed_tensor 的形状为 (N, K, 4, 4)
    
    # 提取平移部分 (x, y, z)
    x = transformed_tensor[..., 0, 3]
    y = transformed_tensor[..., 1, 3]
    z = transformed_tensor[..., 2, 3]
    
    # 计算绕 z 轴的旋转角度 w
    # 角度 w 可以通过 atan2 函数计算
    w = torch.atan2(transformed_tensor[..., 1, 0], transformed_tensor[..., 0, 0])
    
    # 将 x, y, z, w 组合成 (N, K, 4) 的张量
    result = torch.stack((x, y, z, w), dim=-1)
    
    return result

def euler_to_se3(x, y, z, roll, pitch, yaw):
    # 计算绕X轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 计算绕Y轴的旋转矩阵
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # 计算绕Z轴的旋转矩阵
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 组合成旋转矩阵 (ZYX 顺序)
    R = Rz @ Ry @ Rx

    # 构造 SE(3) 矩阵
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = [x, y, z]

    return se3_matrix, R

def transform_state(state):
    
    state_tensor = torch.from_numpy(state)
    state_matrices = transform_matrices_batch(state_tensor)
    # T = torch.tensor([
    #                 [-1.0000000,  0.0000000,  0.0000000, 0.0],
    #                 [-0.0000000,  0.3420202, -0.9396926, 0.1],
    #                 [-0.0000000, -0.9396926, -0.3420202, 0.3],
    #                 [0.0000000, 0.0, 0.0, 1.0]
    #             ], dtype=torch.float64)    
    # x, y, z = 0.0, 0.0, 0.3
    # roll, pitch, yaw = np.radians(-110), np.radians(0), np.radians(-90)  # 转换为弧度 world转转到camera
    
    #scence4
    # x, y, z = -0.25, 0.3, 1.5 
    # roll, pitch, yaw = np.radians(-115), np.radians(0), np.radians(-90)  # 转换为弧度 world转转到camera
    
    #scence5
    x, y, z = -0.18, 0.10, 1.5
    roll, pitch, yaw = np.radians(-119), np.radians(0), np.radians(-90)  # 转换为弧度 world转转到camera

    se3_matrix, so3_matrix = euler_to_se3(x, y, z, roll, pitch, yaw)

    transformed_matrices = apply_transformation(state_matrices, se3_matrix)
    # position_and_w = extract_position_and_angle(transformed_matrices)
    trans = transformed_matrices[:,:,0:3,-1]
    rot = transformed_matrices[:,:,0:3,0:3]

    return trans, rot

# def init_prob_grid(shape,states,target_tf):
#     q_grid = np.zeros(shape)
#     w_grid = np.zeros(shape)
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             basic_similarity = 0
#             for state in states[i][j]:
#                 camera.update(state[:3],rotation_matrix_y(state[3]))
#                 im = renderer.render().unsqueeze(0)
#                 current_similarity = vlad_p.get_vlad_loss(im, target_tf)
#                 q_grid[j][i] += current_similarity
#                 if current_similarity > basic_similarity:
#                     basic_similarity = current_similarity
#                     w_grid[j][i] = state[3]
#     return q_grid, w_grid

def get_id(x,y,shape=(6,6)):
    j = np.floor(3 + x).astype(int)
    i = np.floor(3 - y).astype(int)
    # convert i,j to int or int array
    return i,j

def get_probability_at_coordinates(x, y, probability_grid):
    # 计算栅格索引
    # j = math.floor(2 + x)
    # i = math.floor(3 - y)
    i,j = get_id(x,y,probability_grid.shape)
    # 检查索引是否超出范围
    if 0 <= i < probability_grid.shape[1] and 0 <= j < probability_grid.shape[0]:
        return probability_grid[i,j] # 注意行列顺序
    else:
        return 0  # 超出范围返回0
    
def add_confidence_to_grid(x, y, confidence, probability_grid):
    i,j = get_id(x,y,probability_grid.shape)
    new_grid = probability_grid.copy()
    new_grid[i,j] += confidence
    new_grid = new_grid/np.sum(new_grid)
    return new_grid

def update_probability_grid(x,y,q,probability_grid):
    i,j = get_id(x,y,probability_grid.shape)
    # 检查索引是否超出范围
    # if 0 <= i < probability_grid.shape[1] and 0 <= j < probability_grid.shape[0]:
    #     return probability_grid # 注意行列顺序
    new_prob = probability_grid[i,j]
    new_grid = probability_grid.copy()
    new_grid[i,j] = new_prob*((1-q)/(1-new_prob*q))
    # update the rest of new_grid  by *=(q/1-new_prob*q)
    mask = np.ones(new_grid.shape).astype(bool)
    mask[i,j] = 0
    new_grid[mask] *= (1/(1-new_prob*q))
    return new_grid

def dynamics(state, control):
    """
    simplified kinematic blimp dynamics

    The state of the blimp is q = (x, y,z, w), where (x, y, z) is the position,
    theta is the yaw angle. The control input is (dx,dy,dz,dw)
    The dynamics are given by
      x' = dx
      y' = dy
      z' = dz
      w' = dw
    """
    state = np.array(state)
    x, y, z, theta = state.T if len(state.shape) > 1 else state
    theta = angle_mod(theta)
    vx, vy, w ,h = control.T if len(control.shape) > 1 else control

    dx = control_duration * vx * np.cos(theta) - control_duration * vy * np.sin(theta)
    dy = control_duration * vx * np.sin(theta) + control_duration * vy * np.cos(theta)

    dz = h * 0
    dtheta = control_duration * w
    # state = np.expand_dims(state) if len(state.shape) == 1 else state
    next_state = np.array([x + dx, y + dy, z + dz, angle_mod(theta + dtheta)]).T
    return next_state

def cost2go(controls,state,obstacles=None,width=None,height=None):
    gocosts = 10 * np.ones(controls.shape[0])
    # gocosts += 10*np.sum(np.abs(angle_mod(controls[:,:,3])),axis=1)
    current_i,current_j = get_id(state[0],state[1])
    # calculate all trajectory: N*K*dim which has same shape with controls
    trajs = np.zeros((controls.shape[0],controls.shape[1],4))
    # initial state
    state = state.copy()
    for k in range(controls.shape[1]):
        trajs[:,k,:] = dynamics(state,controls[:,k,:])
        state = trajs[:,k,:]
    if obstacles is not None:
        for obstacle in obstacles:
            # 计算每个状态到所有障碍物的距离
            distances = np.linalg.norm(trajs[:, :, :2][:, :, np.newaxis] - obstacle, axis=3)
            mask = distances < 0.8
            # print(mask.shape)
            gocosts[np.any(mask, axis=(1,2))] += 1000
    if width is not None and height is not None:
        mask1 = np.logical_or(trajs[:,:,0] < width[0], trajs[:,:,0] > width[1])
        mask2 = np.logical_or(trajs[:,:,1] < height[0], trajs[:,:,1] > height[1])
        mask3 = np.logical_or(trajs[:,:,2] < 0.2, trajs[:,:,2] > 0.9)
        mask = np.logical_or(mask1, mask2, mask3)
        gocosts[np.any(mask, axis=1)] += 1000
    # calculate the cost of movement, if the new state id is not the same as the current state id, then add 50, else add 0
    # new_i,new_j = get_id(trajs[:,:,0],trajs[:,:,1]) # N*K*2
    # for k in range(trajs.shape[1]):
    #     mask1 = new_i[:,k] != current_i
    #     mask2 = new_j[:,k] != current_j
    #     mask = np.logical_or(mask1, mask2)
    #     gocosts[mask] += 0
    #     current_i,current_j = new_i[:,k],new_j[:,k]
    return gocosts

def fuzz(control,sigma):
    # control: K*dim
    # print(control.shape)
    fuzzed_control = np.zeros(control.shape)
    non_zero_dims = np.random.randint(3, size=(control.shape[0]))
    random_values_vx = np.random.normal(control[:,0], sigma)
    random_values_vy = np.random.normal(control[:,1], sigma)
    random_values_w = np.random.normal(control[:,2], sigma)
    cols = np.arange(control.shape[0])
    mask = (non_zero_dims == 0)
    fuzzed_control[cols[mask], non_zero_dims[mask]] = random_values_vx[mask]
    mask = (non_zero_dims == 1)
    fuzzed_control[cols[mask], non_zero_dims[mask]] = random_values_vy[mask]
    mask = (non_zero_dims == 2)
    fuzzed_control[cols[mask], non_zero_dims[mask]] = random_values_w[mask]
    # fuzzed_control[:,3] = np.random.normal(0, 0.05, 5)
    return fuzzed_control

    # fuzzed_control[:,:2] = np.clip(fuzzed_control[:,:2], -1, 1)

def close_loop_position_control(desired_state, current_state, control, kp=1.0,max_speed=0.3):
        non_zero_indices = np.where(control != 0)[0]
        v_x, v_y, v_w = 0, 0, 0
        move_cmd = Twist()

        while np.linalg.norm(desired_state - current_state) > 0.03:
            current_state = np.array(ros_interface.acquire_optical_track_pose())
            # 提取目标和当前状态
            x_d, y_d, w_d = desired_state[0], desired_state[1], desired_state[3]
            x_c, y_c, w_c = current_state[0], current_state[1], current_state[3]
            
            # 计算目标点相对于当前状态的世界坐标位移
            dx = x_d - x_c
            dy = y_d - y_c

            # 将世界坐标系下的位移转换到当前状态自身坐标系下
            x_prime = math.cos(w_c) * dx + math.sin(w_c) * dy
            y_prime = -math.sin(w_c) * dx + math.cos(w_c) * dy
            delta_w = angle_mod(w_d - w_c)

            v_x, v_y, v_w = 0, 0, 0
            
            if non_zero_indices == 0:
                v_x = kp * x_prime 
            elif non_zero_indices == 1:
                v_y = kp * y_prime
            elif non_zero_indices == 2:
                v_w = kp * delta_w
                
            v_x = max(min(v_x, max_speed), -max_speed)
            v_y = max(min(v_y, max_speed), -max_speed)
            v_w = max(min(v_w, max_speed), -max_speed)

            move_cmd.linear.x = v_x
            move_cmd.linear.y = v_y
            move_cmd.angular.z = v_w
            ros_interface.pub.publish(move_cmd)

            if abs(v_x) + abs(v_y)  < 0.01 and abs(v_w) < 0.04:
                break

        start_time = time.time()
        move_cmd.linear.x = 0
        move_cmd.linear.y = 0
        move_cmd.angular.z = 0
        while time.time() - start_time < 1.0:
            ros_interface.pub.publish(move_cmd)

class MPPI_controller_cpu():
    def __init__(self,N,K,mu,sigma,state,log=False):
        self.N = N
        self.K = K
        self.mu = np.array(mu) # 1x2
        self.sigma = np.array(sigma) # 1x2
        self.dim = 4
        self.seed = 0
        self.state = state
        self.log = log
        self.count = 0
        # self.probability_grid = np.full((5, 5), 1/25)
        self.probability_grid = np.full((6, 6), 1/36)
        self.current_similarity = 0
        # self.best_similarity = 0.1 # task1
        self.best_similarity = 0.035 # task2
        # self.best_similarity = 0.1 # task3
        # self.probability_grid[1, 3] = 0
        # self.probability_grid[3, 3] = 0

    # def sample_controls(self):
    #     # sample NxK 4-dim vector from normal distribution
    #     np.random.seed(self.seed)
    #     sampled_controls = np.zeros((self.N, self.K, 4))
    #     non_zero_dims = np.random.randint(4, size=(self.N, self.K))
    #     # 生成随机数
    #     random_values_3d = np.random.normal(self.mu[0], self.sigma[0], (self.N, self.K))
    #     random_values_4d = np.random.normal(self.mu[1], self.sigma[1], (self.N, self.K))
    #     # 使用高级索引和广播设置非零元素
    #     rows, cols = np.indices((self.N, self.K))
    #     # 对前三个维度进行赋值
    #     for dim in range(3):
    #         mask = (non_zero_dims == dim)
    #         sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_3d[mask]
    #     # 对第四个维度进行赋值
    #     mask = (non_zero_dims == 3)
    #     sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_4d[mask]
    #     return sampled_controls

    # def sample_controls(self):
    #     np.random.seed(self.seed)
    #     sampled_controls = np.zeros((self.N, self.K, 4))
    #     non_zero_dims = np.random.randint(3, size=(self.N, self.K))
    #     random_values_vx = np.random.normal(self.mu[0], self.sigma[0], (self.N, self.K))
    #     random_values_vy = np.random.normal(self.mu[1], self.sigma[1], (self.N, self.K))
    #     random_values_w = np.random.normal(self.mu[2], self.sigma[2], (self.N, self.K))
    #     sampled_controls[:,:,3] = np.random.normal(0, 0.05, (self.N, self.K))
    #     rows, cols = np.indices((self.N, self.K))
    #     mask = (non_zero_dims == 0)
    #     sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_vx[mask]
    #     mask = (non_zero_dims == 1)
    #     sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_vy[mask]
    #     mask = (non_zero_dims == 2)
    #     sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_w[mask]
    #     return sampled_controls
    
    def sample_controls(self,controls=None,weights=None):
        np.random.seed(self.seed)
        sampled_controls = np.zeros((self.N, self.K, 4))
        if np.all(controls == None):
            # choose explore or exploit randomly at the first step
            non_zero_dims = np.random.randint(3, size=(self.N-8, self.K))
            random_values_vx = np.random.uniform(-0.1, 0.5, (self.N-8, self.K))
            random_values_vy = np.random.normal(self.mu[1], self.sigma[1], (self.N-8, self.K))
            random_values_w = np.random.normal(self.mu[2], self.sigma[2], (self.N-8, self.K))
            # sampled_controls[:,:,3] = np.random.normal(0, 0.05, (self.N, self.K))
            rows, cols = np.indices((self.N-8, self.K))
            mask = (non_zero_dims == 0)
            sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_vx[mask]
            mask = (non_zero_dims == 1)
            sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_vy[mask]
            mask = (non_zero_dims == 2)
            sampled_controls[rows[mask], cols[mask], non_zero_dims[mask]] = random_values_w[mask]
        else:
            controls = np.roll(controls,-1,axis=1)
            index = int(np.random.random()*self.N)
            weighted_max = np.max(weights)
            beta = 0.0
            for i in range(self.N-8):
                beta += np.random.random()*2*weighted_max
                while beta > weights[i]:
                    beta -= weights[i]
                    index = (index + 1) % (self.N)
                # sampled_controls[i] = controls[index]
                if np.random.random() < 0.3:
                    sampled_controls[i] = fuzz(controls[index],max(0,(0.1 - self.current_similarity)*10))
                else:
                    sampled_controls[i] = controls[index]
        sampled_controls[-8:,0,2] = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/4, -np.pi/2, -3*np.pi/4]
        sampled_controls[-8:,1:,0] = max(0,(0.1 - self.current_similarity)*10)
        return sampled_controls


    def rollout(self,controls):
        # calculate the rollout trajectory
        rollout_traj = np.zeros((self.N, self.K+1, self.dim))
        # copy state to the first row of rollout_traj
        state = self.state.copy()
        rollout_traj[:, 0, :] = state
        # N rollout can be calculate as same time
        for k in range(1,self.K+1):
            rollout_traj[:, k, :] = dynamics(state, controls[:, k-1])
            # check if the rollout is out of boundary
            # set the first 2 elements in such rollout to the boundary value
            # rollout_traj[:, k, 0] = np.clip(rollout_traj[:, k, 0], -2+0.1, 3-0.1)
            # rollout_traj[:, k, 1] = np.clip(rollout_traj[:, k, 1], -2+0.1, 3-0.1)
            # rollout_traj[:, k, 2] = np.clip(rollout_traj[:, k, 2], 0.3, 0.8)
            # rollout_traj[:, k, 0] = np.clip(rollout_traj[:, k, 0], -3+0.1, 3-0.1)
            # rollout_traj[:, k, 1] = np.clip(rollout_traj[:, k, 1], -3+0.1, 3-0.1)
            # rollout_traj[:, k, 2] = np.clip(rollout_traj[:, k, 2], 0.0, 0.7)
            rollout_traj[:, k, 0] = np.clip(rollout_traj[:, k, 0], -1.5, 1.5)
            rollout_traj[:, k, 1] = np.clip(rollout_traj[:, k, 1], -1.5, 1.5)
            rollout_traj[:, k, 2] = np.clip(rollout_traj[:, k, 2], 0.0, 0.7)
            rollout_traj[:, k, 3] = angle_mod(rollout_traj[:, k, 3])
            state = rollout_traj[:, k, :]
        if self.log:
            # save rollout_traj
            np.save(f'temp/{task_name}/rollout_traj_{self.count:04d}.npy', rollout_traj)
        return rollout_traj

    # def update(self,state,mu=None,sigma=None,K=None):
    #     self.state = state
    #     if mu is not None:
    #         self.mu = mu
    #         self.sigma = sigma
    #     if K is not None:
    #         self.K = K
    #     self.count += 1

    def predict(self,controls):
        rollout_traj = self.rollout(controls) # N*K+1*4 start from self.state
        # traj_cost = cost2go(controls, self.state,width=np.array([-2,3]),height=np.array([-2,3]),obstacles=[np.array([1.36, 1.31]),np.array([1.3, -0.8]),np.array([0.05, 0])])
        traj_cost = cost2go(controls, self.state,width=np.array([-1.5,1.5]),height=np.array([-1.5,1.5]),obstacles=[np.array([0.7,0])])#,np.array([0.5, 0]),np.array([1.4, 1.5]),np.array([0,1.75])])
        trans, rot = transform_state(rollout_traj[:,1:,:])
        image_similarity = np.zeros((self.N)) # 10位数，基本上是20~30
        bayesian_prob = np.zeros((self.N)) # 0~1
        for i,traj in enumerate(trans):
            #traj: K*dim
             for j,state in enumerate(traj):
                camera.update(state,rot[i,j,:,:])
                im = renderer.render().unsqueeze(0)
                if j < self.K-1:
                    image_similarity[i] += vlad_s.get_vlad_loss(im, tf_target)
                else:
                    image_similarity[i] += 100 if vlad_p.get_vlad_loss(im, tf_target) > self.current_similarity else 0
                bayesian_prob[i] += 50*get_probability_at_coordinates(rollout_traj[i,j+1,0],rollout_traj[i,j+1,1],self.probability_grid)
        # calculate the cost
        print("costs:",traj_cost[-8:])
        print("image_similarity:",image_similarity[-8:])
        print("bayesian_prob:",bayesian_prob[-8:])
        weights = (image_similarity*bayesian_prob)/traj_cost
        max_index = np.argmax(weights)
        print("best costs:",traj_cost[max_index])
        print("best_image_similarity:",image_similarity[max_index])
        print("best_bayesian_prob:",bayesian_prob[max_index])
        control = controls[max_index,0,:]
        # control = np.average(controls[:,0,:],axis=0,weights=weights)
        if self.log:
            np.save(f'temp/{task_name}/control_{self.count:04d}.npy', control)
        return control, weights
    
    def update(self,control):   
        print('control: ', control)

        if RUN_SIM: # sim
            
            desired_state = np.hstack(dynamics(self.state,control))
            print('desired_state: ', desired_state)

            # Update new_state with dynamics directly
            new_state = desired_state 
            
            if OFFLINE_MODE: # for offline control and visualization
                current_state = np.array(ros_interface.acquire_optical_track_pose())
                print("current_state: ", current_state)

                # close loop control
                close_loop_position_control(desired_state=desired_state, current_state=current_state, control=control)

                current_state = np.array(ros_interface.acquire_optical_track_pose())
                print("moved_to: ", current_state)

                # Real camera visualization
                cv_im = ros_interface.acquire_color_image()
                rgb_image = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB) # 输出形状为 (H, W, C)
                normalized_image = rgb_image / 255.0
                tensor_image = torch.from_numpy(normalized_image).float()
                im = tensor_image.permute(2, 0, 1).unsqueeze(0)  # 调整为 (1, C, H, W)，以匹配深度学习框架的格式

                current_view = im.clone().cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
                current_view = (current_view * 255).astype(np.uint8)
                current_view = PIL.Image.fromarray(current_view)
                current_view.save(f'temp/{task_name}/camera_view_output_{self.count}.png')
                plt.figure("Camera View")
                plt.imshow(current_view)
                plt.show(block=False)
                plt.pause(1)

        else: # online mode
            desired_state = np.hstack(dynamics(self.state,control))
            print("desired_state: ", desired_state)

            current_state = np.array(ros_interface.acquire_optical_track_pose())
            print("current_state: ", current_state)
                        
            # close loop control
            close_loop_position_control(desired_state=desired_state, current_state=current_state, control=control)

            # Update new_state with optical track
            new_state = np.array(ros_interface.acquire_optical_track_pose())
            print("moved to: ", new_state)

            # Real camera visualization
            cv_im = ros_interface.acquire_color_image()
            rgb_image = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB) # 输出形状为 (H, W, C)
            normalized_image = rgb_image / 255.0
            tensor_image = torch.from_numpy(normalized_image).float()
            im = tensor_image.permute(2, 0, 1).unsqueeze(0)  # 调整为 (1, C, H, W)，以匹配深度学习框架的格式

            current_view = im.clone().cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
            current_view = (current_view * 255).astype(np.uint8)
            current_view = PIL.Image.fromarray(current_view)
            current_view.save(f'temp/{task_name}/camera_view_output_{self.count}.png')
            plt.figure("Camera View")
            plt.imshow(current_view)
            plt.show(block=False)
            plt.pause(1)

            # move_cmd = Twist()
            # move_cmd.linear.x = control[0]
            # move_cmd.linear.y = control[1]
            # move_cmd.angular.z = control[2]
            # start_time = time.time()
            # while time.time() - start_time < control_duration:
            #     ros_interface.pub.publish(move_cmd)
            # move_cmd.linear.x = 0
            # move_cmd.linear.y = 0
            # move_cmd.angular.z = 0
            # while time.time() - start_time < control_duration + 1.0:
            #     ros_interface.pub.publish(move_cmd)

            # # new_state = np.array(ros_interface.acquire_odom())
            # new_state = np.array(ros_interface.acquire_optical_track_pose())
            # print('odom: ', new_state)
        
        # Render camera visualization
        trans, rot = transform_state(np.array([[new_state]]))
        camera.update(trans[0,0,:], rot[0,0,:,:])
        im = renderer.render().unsqueeze(0) # (1, C, H, W) 在维度 0 位置增加一个新的维度，因此 im 的形状从 (C, H, W) 变为 (1, C, H, W)。这种形状是典型的 batch 处理格式，方便在深度学习中处理多张图片的情形。
        render_view = im.clone().cpu().squeeze(0).permute(1, 2, 0).detach().numpy() # 将通道顺序从 (C, H, W) 变为 (H, W, C)，使其符合 NumPy 和大部分图像库（如 PIL 和 Matplotlib）对图像的通道顺序要求，即高度、宽度、通道顺序
        render_view = (render_view * 255).astype(np.uint8)
        render_view = PIL.Image.fromarray(render_view) # Matplotlib 处理的图像格式为 (H, W, C)
        render_view.save(f'temp/{task_name}/render_view_output_{self.count}.png')
        plt.figure("Render View")
        plt.imshow(render_view)
        plt.show(block=False)
        plt.pause(1)

        current_similarity = vlad_p.get_vlad_loss(im, tf_target)
        print(current_similarity)
        if current_similarity > 0.035:#0.08:
            if current_similarity > self.best_similarity:
                self.best_similarity = current_similarity
                print("Find the target!")
                self.probability_grid = add_confidence_to_grid(new_state[0],new_state[1],current_similarity,self.probability_grid)
        else:
            self.probability_grid = update_probability_grid(new_state[0],new_state[1],current_similarity,self.probability_grid)
        if self.log:
            np.save(f'temp/{task_name}/prob_grid_{self.count:04d}.npy', self.probability_grid)
        self.state = new_state
        self.mu = (control[0],control[1])
        self.sigma = (1,0.3)
        self.count += 1
        self.current_similarity = current_similarity

def main():
    # initial state
    if RUN_SIM or OFFLINE_MODE:
        state = [-1.4, -0.7, 0.0, 1.5776]
        # state = [-1.4, 0.0, 0.0, 0.0]
    else:
        state = ros_interface.acquire_optical_track_pose()
    states =[state]
    controller = MPPI_controller_cpu(N=100, K=5, mu=(0,0,0), sigma=(0.3,0.3,0.1),state=state,log=True)
    # while not Found:
    controls = None
    weights = None
    for i in tqdm(range(150)):
        controls = controller.sample_controls(controls,weights)
        control,weights = controller.predict(controls)
        controller.update(control)
        states.append(controller.state)

if __name__ == "__main__":
    main()
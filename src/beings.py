import numpy as np
from utils import angle_mod
import os
import sys
# 获取当前文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加3D文件夹的路径
sys.path.append(os.path.join(current_dir, '..', '3DGS_PoseRender'))

from tqdm import tqdm
from PIL import Image
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer

from utils import rotation_matrix_y

import torchvision.transforms as transforms

from vlad_loss import VLAD_SIM
vlad_s = VLAD_SIM("patchnetvlad_root/patchnetvlad/configs/speed.ini")
vlad_p = VLAD_SIM("patchnetvlad_root/patchnetvlad/configs/performance.ini")



model_path = "scence\\01\in_paper.ply" # Path to the ply file model
camera = Camera()
camera_info = {'width': 1920,
                'height': 1440,
                'position': [0, -0.5, 0],
                'rotation': [[1,0,0],[0,1,0],[0,0,1]],
                'fy': 1371.7027360999618,
                'fx': 1371.7027360999618}
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
task_name = "hard"
target = Image.open(f"target/scence01/{task_name}.jpg")
it = input_transform((360,480))
tf_target = it(target).unsqueeze(0).cuda()
# if ./PIE_temp/task_name not exist, create it
if not os.path.exists(f'./PIE_temp/{task_name}'):
    os.makedirs(f'./PIE_temp/{task_name}')

def transform_state(state):
    # new_state has same shape with state
    new_state = state.copy()
    # rotate the state 180 degree along y-axis
    new_state[:,:,:3] = -state[:,:,:3]
    # switch new_state[:,:,1] and new_state[:,:,2]
    new_state[:,:,1], new_state[:,:,2] = new_state[:,:,2].copy(), new_state[:,:,1].copy()
    new_state[:,:,3] = -(state[:,:,3]+np.pi/2)
    return new_state

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
    dx = vx * np.cos(theta) + vy * np.sin(theta)
    dy = vx * np.sin(theta) - vy * np.cos(theta)
    dz = h
    dtheta = w
    # state = np.expand_dims(state) if len(state.shape) == 1 else state
    next_state = np.array([x + dx, y + dy, z + dz, angle_mod(theta + dtheta)]).T
    return next_state

def cost2go(controls,state,obstacles=None,width=None,height=None):
    gocosts = 10*np.ones(controls.shape[0])
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
            mask = distances < 0.5
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
    fuzzed_control[:,3] = np.random.normal(0, 0.05, 5)
    return fuzzed_control

    # fuzzed_control[:,:2] = np.clip(fuzzed_control[:,:2], -1, 1)

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
            sampled_controls[:,:,3] = np.random.normal(0, 0.05, (self.N, self.K))
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
            rollout_traj[:, k, 0] = np.clip(rollout_traj[:, k, 0], -3+0.1, 3-0.1)
            rollout_traj[:, k, 1] = np.clip(rollout_traj[:, k, 1], -3+0.1, 3-0.1)
            rollout_traj[:, k, 2] = np.clip(rollout_traj[:, k, 2], 0.0, 0.7)
            rollout_traj[:, k, 3] = angle_mod(rollout_traj[:, k, 3])
            state = rollout_traj[:, k, :]
        if self.log:
            # save rollout_traj
            np.save(f'PIE_temp/{task_name}/rollout_traj_{self.count:04d}.npy', rollout_traj)
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
        traj_cost = cost2go(controls, self.state,width=np.array([-3,3]),height=np.array([-3,3]),obstacles=[np.array([-0.5,0]),np.array([0.5, 0]),np.array([1.4, 1.5]),np.array([0,1.75])])
        render_traj = transform_state(rollout_traj[:,1:,:])
        image_similarity = np.zeros((self.N)) # 10位数，基本上是20~30
        bayesian_prob = np.zeros((self.N)) # 0~1
        for i,traj in enumerate(render_traj):
            #traj: K*dim
             for j,state in enumerate(traj):
                camera.update(state[:3],rotation_matrix_y(state[3]))
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
            np.save(f'PIE_temp/{task_name}/control_{self.count:04d}.npy', control)
        return control, weights
    
    def update(self,control):
        # print(self.state)
        # print(control)
        new_state = dynamics(self.state,control)
        # print(new_state)
        new_render_state = transform_state(new_state.reshape(1,1, 4)).reshape(4,)
        camera.update(new_render_state[:3],rotation_matrix_y(new_render_state[3]))
        im = renderer.render().unsqueeze(0)
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
            np.save(f'PIE_temp/{task_name}/prob_grid_{self.count:04d}.npy', self.probability_grid)
        self.state = new_state
        self.mu = (control[0],control[1])
        self.sigma = (1,0.3)
        self.count += 1
        self.current_similarity = current_similarity

def main():
    # initial state
    state = [2, 2, 0.3, np.pi]
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
import plotly.graph_objs as go
import numpy as np
import os
from PIL import Image  # 用于将图片合成为 GIF
# 示例4DOF坐标数据
keywords = "hard"
data_path = f".\\result\\states\\{keywords}"
# laod every .npy file in the data_path

states_list = []
names = []

for file in os.listdir(data_path):
    if file.endswith(".npy"):
        # load file as the file name oder a numpy array
        states = np.load(os.path.join(data_path, file))
        name = os.path.splitext(file)[0]
        names.append(name)
        states_list.append(states)

# 示例4DOF坐标数据
coords = np.array(states_list[2])
# Target位置
target = np.array([-1.0, 0, 0.5, np.pi])  # [x, y, z, yaw]

# 提取x, y, z和yaw
x, y, z, yaw = coords[:, 1], -coords[:, 0], coords[:, 2], coords[:, 3]+(np.pi/2)

# 计算yaw方向的向量（长度设定为1）
def yaw_to_vector(yaw, length=1):
    return np.array([np.cos(yaw) * length, np.sin(yaw) * length, 0])

# 起始点和Target的yaw向量
start_vec = yaw_to_vector(yaw[0])
target_vec = yaw_to_vector(target[3])

# 创建轨迹线
trace_line = go.Scatter3d(x=x, y=y, z=z, mode='lines', name='轨迹',line=dict(color='blue', width=10))

# 创建起始点的yaw金字塔
start_pyramid = go.Cone(
    x=[x[0]], y=[y[0]], z=[z[0]],
    u=[start_vec[0]], v=[start_vec[1]], w=[start_vec[2]],
    sizemode="scaled", sizeref=0.5, anchor="tip", name="起始yaw",
    colorscale='Blues'
)

# 创建target的yaw金字塔
target_pyramid = go.Cone(
    x=[target[0]], y=[target[1]], z=[target[2]],
    u=[target_vec[0]], v=[target_vec[1]], w=[target_vec[2]],
    sizemode="scaled", sizeref=0.5, anchor="tip", name="Target",
    colorscale='Greens'
)

# 添加圆柱体障碍物1
cylinder1_height = 1.3
cylinder1_radius = 0.2
cylinder1_y = -1.36
cylinder1_x = 1.31
cylinder1_z = np.linspace(0, cylinder1_height, 20)
theta = np.linspace(0, 2 * np.pi, 50)
theta_grid, z_grid = np.meshgrid(theta, cylinder1_z)
x_cylinder1 = cylinder1_radius * np.cos(theta_grid) + cylinder1_x
y_cylinder1 = cylinder1_radius * np.sin(theta_grid) + cylinder1_y
z_cylinder1 = z_grid

# 添加圆柱体障碍物2
cylinder2_height = 1.3
cylinder2_radius = 0.2
cylinder2_y = -1.3
cylinder2_x = -0.8
cylinder2_z = np.linspace(0, cylinder2_height, 20)
theta = np.linspace(0, 2 * np.pi, 50)
theta_grid, z_grid = np.meshgrid(theta, cylinder2_z)
x_cylinder2 = cylinder2_radius * np.cos(theta_grid) + cylinder2_x
y_cylinder2 = cylinder2_radius * np.sin(theta_grid) + cylinder2_y
z_cylinder2 = z_grid

# 添加圆柱体障碍物3
cylinder3_height = 1.0
cylinder3_radius = 0.2
cylinder3_y = 0.0
cylinder3_x = 0
cylinder3_z = np.linspace(0, cylinder3_height, 20)
theta = np.linspace(0, 2 * np.pi, 50)
theta_grid, z_grid = np.meshgrid(theta, cylinder3_z)
x_cylinder3 = cylinder3_radius * np.cos(theta_grid) + cylinder3_x
y_cylinder3 = cylinder3_radius * np.sin(theta_grid) + cylinder3_y
z_cylinder3 = z_grid



cylinder1_surface = go.Surface(
    x=x_cylinder1,
    y=y_cylinder1,
    z=z_cylinder1,
    colorscale=[[0, 'black'], [1, 'black']],
    showscale=False,
    name='障碍物1'
)

cylinder2_surface = go.Surface(
    x=x_cylinder2,
    y=y_cylinder2,
    z=z_cylinder2,
    colorscale=[[0, 'black'], [1, 'black']],
    showscale=False,
    name='障碍物2'
)

cylinder3_surface = go.Surface(
    x=x_cylinder3,
    y=y_cylinder3,
    z=z_cylinder3,
    colorscale=[[0, 'black'], [1, 'black']],
    showscale=False,
    name='障碍物3'
)

# 初始化帧列表
frames = []

# 创建每一帧，逐步增长轨迹线
for i in range(1, len(x)):
    # 当前帧的yaw金字塔
    current_vec = yaw_to_vector(yaw[i])
    current_pyramid = go.Cone(
        x=[x[i]], y=[y[i]], z=[z[i]],
        u=[current_vec[0]], v=[current_vec[1]], w=[current_vec[2]],
        sizemode="scaled", sizeref=0.5, anchor="tip", name="当前yaw",
        colorscale='Oranges'
    )

    # 更新轨迹线，包含到当前帧为止的所有点
    trace_line_frame = go.Scatter3d(x=x[:i+1], y=y[:i+1], z=z[:i+1], mode='lines', name='轨迹')

    # 在每一帧中，保持轨迹、起始yaw、target、当前yaw和障碍物
    frames.append(go.Frame(data=[trace_line_frame, start_pyramid, target_pyramid, current_pyramid, cylinder1_surface,cylinder2_surface,cylinder3_surface], name=f'frame{i}'))

# 创建图表布局
layout = go.Layout(
    scene=dict(
        xaxis=dict(range=[-3, 3], title='Y'),
        yaxis=dict(range=[-3, 3], title='X'),
        zaxis=dict(range=[0, 3], title='Z'),
        aspectmode='cube'
    ),
    title="4DOF轨迹与Yaw角动画",
    updatemenus=[dict(type="buttons", showactive=False,
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])]
)

# 创建图表并添加布局和帧
fig = go.Figure(data=[trace_line, start_pyramid, target_pyramid, current_pyramid,  cylinder1_surface,cylinder2_surface,cylinder3_surface], layout=layout, frames=frames)

# 显示图表
fig.show()
# save the animation to gif
# 导出为GIF动画
if not os.path.exists("frames"):
    os.makedirs("frames")

# 保存每一帧
for i, frame in enumerate(frames):
    fig.update(frames=[frame])  # 更新到当前帧
    fig.write_image(f"frames/frame_{i:02d}.png")

# 使用Pillow将帧合成为GIF
# frames = [Image.open(f"frames/frame_{i:02d}.png") for i in range(len(frames))]
# frames[0].save('animation.gif', save_all=True, append_images=frames[1:], optimize=False, duration=500, loop=0)

# # 清理临时帧文件
# for i in range(len(frames)):
#     os.remove(f"frames/frame_{i:02d}.png")

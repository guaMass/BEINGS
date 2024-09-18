# BEINGS: Bayesian Embodied Image-goal Navigation with Gaussian Splatting
This repository includes the code that can be used to reproduce the results of our paper [BEINGS: Bayesian Embodied Image-goal Navigation with Gaussian Splatting](https://arxiv.org/abs/2409.10216)

## Clone the repo
```
git clone https://github.com/guaMass/BEINGS.git --recursive
```

## Download scences
Download scences from our [website](https://www.mwg.ink/BEINGS-web/), and put them in scence folder. Or just unzip the zip file in the project's root path.

## Enviroment configuration
Create a new conda enviroment
```
conda create -n beings python=3.10 
conda activate beings
```
Install CUDA and Visual studio, add them to PATH.
Install Pytorch
```
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
Install requirments

## Run
Edit `./src/beings.py` before you running.
+ model_path: Use the `.ply` path of scence folder.
+ task_name: Use the target image that you what the robot find.

Run `./src/beings.py`.

## Visualization
There are two ways to view the results of BEINGS. If you choose to log while BEINGS is running, you can watch the results at each time step in real time via the notebook. Or at the end of a BEINGS run, view the animation of the run via `./src/visual_animation_3d.py`.

## Run with your own map
Feel free to try BEINGS on your own 3DGS file (must be `.ply` formmat) and your own images.

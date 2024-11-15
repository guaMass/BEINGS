import sys
import os

# 获取当前文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加patch/patchv目录到sys.path
sys.path.append(os.path.join(current_dir, '..', 'patchnetvlad_root'))

from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools.patch_matcher import PatchMatcher
# from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.models.local_matcher import calc_keypoint_centers_from_patches as calc_keypoint_centers_from_patches
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

import torch
import torchvision.transforms as transforms
import numpy as np
import configparser
from os.path import join

# configfile = "D:\Project\ICRA\Patch-NetVLAD\patchnetvlad\configs\performance.ini"
# config = configparser.ConfigParser()
# config.read(configfile)

# PATCHNETVLAD_ROOT_DIR = os.path.join(current_dir, '..', 'patchnetvlad_root')
# print(PATCHNETVLAD_ROOT_DIR)
class VLAD_SIM():
    def __init__(self,configfile):
        self.config = configparser.ConfigParser()
        self.config.read(configfile)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_dim, self.encoder = get_backend()
        self.resume_ckpt = join(PATCHNETVLAD_ROOT_DIR, self.config['global_params']['resumePath'] + self.config['global_params']['num_pcs'] + '.pth.tar')
        self.checkpoint = torch.load(self.resume_ckpt, map_location=lambda storage, loc: storage)
        self.config['global_params']['num_clusters'] = str(self.checkpoint['state_dict']['pool.centroids'].shape[0])
        self.model = get_model(self.encoder, self.encoder_dim, self.config['global_params'], append_pca_layer=True)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(self.device)
        self.patch_sizes = [int(s) for s in self.config['global_params']['patch_sizes'].split(",")]
        self.strides = [int(s) for s in self.config['global_params']['strides'].split(",")]
        self.patch_weights = np.array(self.config['feature_match']['patchWeights2Use'].split(",")).astype(float)


    def get_vlad_loss(self, im1, im2):
        input_data = torch.cat((im1.to(self.device), im2.to(self.device)), 0)
        with torch.no_grad():
            image_encoding = self.model.encoder(input_data)

            vlad_local, _ = self.model.pool(image_encoding)
            # global_feats = get_pca_encoding(model, vlad_global).cpu().numpy()

            local_feats_one = []
            local_feats_two = []
            for this_iter, this_local in enumerate(vlad_local):
                this_local_feats = get_pca_encoding(self.model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
                    reshape(this_local.size(2), this_local.size(0), int(self.config['global_params']['num_pcs'])).permute(1, 2, 0)
                local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))
                local_feats_two.append(this_local_feats[1, :, :])
            
        all_keypoints = []
        all_indices = []
        for patch_size, stride in zip(self.patch_sizes, self.strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
            keypoints, indices = calc_keypoint_centers_from_patches(self.config['feature_match'], patch_size, patch_size, stride, stride)
            all_keypoints.append(keypoints)
            all_indices.append(indices)
        
        matcher = PatchMatcher(self.config['feature_match']['matcher'], self.patch_sizes, self.strides, all_keypoints,
                           all_indices)
        scores, inlier_keypoints_one, inlier_keypoints_two = matcher.match(local_feats_one, local_feats_two)
        score = -apply_patch_weights(scores, len(self.patch_sizes), self.patch_weights)
        return score

def apply_patch_weights(input_scores, num_patches, patch_weights):
    output_score = 0
    if len(patch_weights) != num_patches:
        raise ValueError('The number of patch weights must equal the number of patches used')
    for i in range(num_patches):
        output_score = output_score + (patch_weights[i] * input_scores[i])
    return output_score
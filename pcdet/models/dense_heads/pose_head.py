import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoseHead(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        
        # Define pose estimation dimensions (position + heading)
        self.pose_dims = 4  # 3 for position (x,y,z) + 1 for heading (yaw angle)
        
        # Feature extraction and regression layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Pose regression
        self.pose_regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, self.pose_dims)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data_dict):
        """
        Args:
            data_dict: Contains spatial_features_2d with shape [B, C, H, W]
        Returns:
            data_dict: With added pose prediction
        """
        # Extract spatial features shared with detection head
        spatial_features_2d = data_dict['spatial_features_2d']  # [B, 384, 248, 216]
        
        # Process features
        features = self.feature_extractor(spatial_features_2d)  # [B, 128, 248, 216]
        
        # Global pooling
        pooled_features = self.global_pool(features)  # [B, 128, 1, 1]
        batch_size = pooled_features.shape[0]
        pooled_features = pooled_features.view(batch_size, -1)  # [B, 128]
        
        # Pose regression
        pose_pred = self.pose_regressor(pooled_features)  # [B, 4]
        
        # Store predictions - position (xyz) and heading (yaw)
        data_dict['batch_ego_pose'] = pose_pred
        
        return data_dict
    
    def get_pose_loss(self, batch_dict, tb_dict=None):
        if tb_dict is None:
            tb_dict = {}
            
        pose_pred = batch_dict['batch_ego_pose']
        pose_gt = batch_dict['ego_pose_gt']  # Should be provided by dataset
        
        # Position loss (L1)
        pos_loss = F.smooth_l1_loss(pose_pred[:, :3], pose_gt[:, :3])
        
        # Heading loss - handle angular wrapping
        heading_pred = pose_pred[:, 3]
        heading_gt = pose_gt[:, 3]
        
        # Compute angular difference and handle wrapping
        heading_diff = heading_pred - heading_gt
        heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
        heading_loss = F.smooth_l1_loss(heading_diff, torch.zeros_like(heading_diff))
        
        # Combined loss
        pose_loss = pos_loss + self.model_cfg.get('HEADING_WEIGHT', 1.0) * heading_loss
        
        tb_dict['pose_position_loss'] = pos_loss.item()
        tb_dict['pose_heading_loss'] = heading_loss.item()
        tb_dict['pose_loss'] = pose_loss.item()
        
        return pose_loss, tb_dict
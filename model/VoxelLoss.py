
from cv2 import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from utils.utils import *

class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, true_label,rm,psm):
        """_summary_
        This function calucate the losses of classication and regression for RPN.
        Args:
            true_label (list) : Ground truth
            rpn_prob_output (tensor): _description_
            delta_output (tensor): _description_

        Returns:
            accumulate_loss (tensor) : total loss = classification loss + regression loss
            cls_loss(tensor) : classification loss
            reg_loss(tensor) : regression loss loss
            cls_pos_loss(tensor) : classication postion loss    
            cls_neg_loss(tensor) :classifcation negative loss
            
        """
        
        self.rpn_output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
        # Calculate ground-truth
        pos_equal_one, neg_equal_one, targets = cal_rpn_target( 
            true_label, self.rpn_output_shape, self.anchors, cls = cfg.DETECT_OBJ, coordinate = 'lidar') 
        
                
        p_pos = F.sigmoid(psm.permute(0,2,3,1))
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),-1,7)
        targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7)

        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        
        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        
        accumulate_loss = conf_loss + reg_loss
        
        return accumulate_loss, conf_loss, reg_loss ,cls_pos_loss, cls_neg_loss
        

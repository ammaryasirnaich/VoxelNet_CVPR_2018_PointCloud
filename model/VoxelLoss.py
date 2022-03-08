
from cv2 import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from utils.utils import *

class VoxelLoss(nn.Module):
    def __init__(self, alpha = 1.5, beta = 1, sigma = 3):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=None)
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
         # Generate anchors
        self.anchors = cal_anchors()    # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 7]; 2 means two rotations; 7 means (cx, cy, cz, h, w, l, r)
        self.small_addon_for_BCE = 1e-6

    def forward(self, true_label,psm,rm):
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
        
        
        pos_equal_one_for_reg = np.concatenate(
            [np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis = -1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis = (1, 2, 3)).reshape(-1, 1, 1, 1), a_min = 1, a_max = None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis = (1, 2, 3)).reshape(-1, 1, 1, 1), a_min = 1, a_max = None)

        # Move to gpu
        pos_equal_one = torch.from_numpy(pos_equal_one).float().cuda()
        neg_equal_one = torch.from_numpy(neg_equal_one).float().cuda()
        targets = torch.from_numpy(targets).float().cuda()
        pos_equal_one_for_reg = torch.from_numpy(pos_equal_one_for_reg).float().cuda()
        pos_equal_one_sum = torch.from_numpy(pos_equal_one_sum).float().cuda()
        neg_equal_one_sum = torch.from_numpy(neg_equal_one_sum).float().cuda()

        # [batch, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2/14] -> [batch, 2/14, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
        pos_equal_one = pos_equal_one.permute(0, 3, 1, 2)
        neg_equal_one = neg_equal_one.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        pos_equal_one_for_reg = pos_equal_one_for_reg.permute(0, 3, 1, 2)

        # Calculate loss
        cls_pos_loss = (-pos_equal_one * torch.log(psm + self.small_addon_for_BCE)) / pos_equal_one_sum
        cls_neg_loss = (-neg_equal_one * torch.log(1 - psm + self.small_addon_for_BCE)) / neg_equal_one_sum

        conf_loss = torch.sum(self.alpha * cls_pos_loss + self.beta * cls_neg_loss)
        cls_pos_loss_rec = torch.sum(cls_pos_loss)
        cls_neg_loss_rec = torch.sum(cls_neg_loss)

        ## using author defined loss function
        reg_loss = smooth_l1(rm * pos_equal_one_for_reg, targets * pos_equal_one_for_reg, self.sigma) / pos_equal_one_sum
        reg_loss = torch.sum(reg_loss)

        accumulate_loss = conf_loss + reg_loss
        
        return accumulate_loss, conf_loss, reg_loss , cls_pos_loss_rec, cls_neg_loss_rec
        


def smooth_l1(deltas, targets, sigma = 3.0):
    # Reference: https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf
    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.lt(torch.abs(diffs), 1.0 / sigma2).float()

    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + torch.mul(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1
import os

import torch
import torch.nn as nn

from config import cfg
from utils.utils import *
from utils.colorize import colorize
from utils.nms import nms
from model.group_pointcloud import FeatureNet
from model.middle_rpn import MiddleAndRPN

import pdb


small_addon_for_BCE = 1e-6


class RPN3D(nn.Module):
    def __init__(self, cls = 'Car', alpha = 1.5, beta = 1, sigma = 3):
        super(RPN3D, self).__init__()

        self.cls = cls
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.feature = FeatureNet()
        self.rpn = MiddleAndRPN()

        # Generate anchors
        self.anchors = cal_anchors()    # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 7]; 2 means two rotations; 7 means (cx, cy, cz, h, w, l, r)

        self.rpn_output_shape = self.rpn.output_shape


    def forward(self, vox_feature, vox_coordinate):
        """_summary_

        Args:
            vox_feature (_type_): _description_
            vox_coordinate (_type_): _description_

        Returns:
            pcm(tensor): Probability score map
            rm(tensor):  Regression map
        """        
        features = self.feature(vox_feature, vox_coordinate)
        pcm, rm = self.rpn(features)
        return pcm, rm
  
       

    def predict(self, data, probs, deltas, summary = False, vis = False):
        '''
        probs: (batch, 2, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH)
        deltas: (batch, 14, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH)
        '''
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        img = data[5]
        lidar = data[6]

        batch_size, _, _, _ = probs.shape
        device = probs.device

        batch_gt_boxes3d = None
        if summary or vis:
            batch_gt_boxes3d = label_to_gt_box3d(label, cls = self.cls, coordinate = 'lidar')

        # Move to cpu and convert to numpy array
        probs = probs.cpu().detach().numpy()
        deltas = deltas.cpu().detach().numpy()

        # BOTTLENECK
        batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate = 'lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        batch_probs = probs.reshape((batch_size, -1))

        # NMS
        ret_box3d = []
        ret_score = []
        for batch_id in range(batch_size):
            # Remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate = 'lidar'))

            # 2D box index after nms
            ind, cnt = nms(torch.from_numpy(boxes2d).to(device), torch.from_numpy(tmp_scores).to(device),
                           cfg.RPN_NMS_THRESH, cfg.RPN_NMS_POST_TOPK)
            
            ind = ind[:cnt].cpu().detach().numpy()

            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis = -1))

        if summary:
            # Only summry the first one in a batch
            cur_tag = tag[0]
            P, Tr, R = load_calib(os.path.join(cfg.CALIB_DIR, cur_tag + '.txt'))

            front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0],
                                                    batch_gt_boxes3d[0], P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

            bird_view = lidar_to_bird_view_img(lidar[0], factor = cfg.BV_LOG_FACTOR)

            bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0], batch_gt_boxes3d[0],
                                                     factor = cfg.BV_LOG_FACTOR, P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

            heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)

            ret_summary = [['predict/front_view_rgb', front_image[np.newaxis, ...]],  # [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3]
                           # [None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR * cfg.INPUT_WIDTH, 3]
                           ['predict/bird_view_lidar', bird_view[np.newaxis, ...]],
                           # [None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3]
                           ['predict/bird_view_heatmap', heatmap[np.newaxis, ...]]]

            return tag, ret_box3d_score, ret_summary

        if vis:
            front_images, bird_views, heatmaps = [], [], []
            for i in range(len(img)):
                cur_tag = tag[i]
                P, Tr, R = load_calib(os.path.join(cfg.CALIB_DIR, cur_tag + '.txt'))

                front_image = draw_lidar_box3d_on_image(img[i], ret_box3d[i], ret_score[i],
                                                        batch_gt_boxes3d[i], P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

                bird_view = lidar_to_bird_view_img(lidar[i], factor = cfg.BV_LOG_FACTOR)

                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i], batch_gt_boxes3d[i],
                                                         factor = cfg.BV_LOG_FACTOR, P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

                heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)

                front_images.append(front_image)
                bird_views.append(bird_view)
                heatmaps.append(heatmap)

            return tag, ret_box3d_score, front_images, bird_views, heatmaps

        return tag, ret_box3d_score


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
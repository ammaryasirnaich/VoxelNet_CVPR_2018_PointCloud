import torch
import torch.nn as nn

from config import cfg

import pdb




class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFELayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = int(out_channels / 2)

        self.dense = nn.Sequential(nn.Linear(self.in_channels, self.units), nn.ReLU())
        self.batch_norm = nn.BatchNorm1d(self.units)


    def forward(self, inputs, mask):
        # [ΣK, T, in_ch] -> [ΣK, T, units] -> [ΣK, units, T]
        tmp = self.dense(inputs).transpose(1, 2)
        # [ΣK, units, T] -> [ΣK, T, units]
        pointwise = self.batch_norm(tmp).transpose(1, 2)

        # [ΣK, 1, units]
        aggregated, _ = torch.max(pointwise, dim = 1, keepdim = True)

        # [ΣK, T, units]
        repeated = aggregated.expand(-1, cfg.VOXEL_POINT_COUNT, -1)

        # [ΣK, T, 2 * units]
        concatenated = torch.cat([pointwise, repeated], dim = 2)

        # [ΣK, T, 1] -> [ΣK, T, 2 * units]
        mask = mask.expand(-1, -1, 2 * self.units)

        concatenated = concatenated * mask.float()

        return concatenated



class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.vfe1 = VFELayer(7, 32)
        self.vfe2 = VFELayer(32, 128)


    def forward(self, feature, coordinate):

        batch_size = len(feature)
        
        # the feature is of list type 
        # print("Feature dimensions before torch.cat", type(feature[0]))  
        # print("Feature dimensions before torch.cat", feature[0].shape)

        feature = torch.cat(feature, dim = 0)   # [ΣK, cfg.VOXEL_POINT_COUNT, 7]; cfg.VOXEL_POINT_COUNT = 35/45
        coordinate = torch.cat(coordinate, dim = 0)     # [ΣK, 4]; each row stores (batch, d, h, w)

        vmax, _ = torch.max(feature, dim = 2, keepdim = True)
        mask = (vmax != 0)  # [ΣK, T, 1]

        x = self.vfe1(feature, mask)
        x = self.vfe2(x, mask)

        # [ΣK, 128]
        voxelwise, _ = torch.max(x, dim = 1)
        
  
         # [ΣK, 138]
        # Appending voxel intensity historgram feature 
        
        # print("Feature dimensions after torch.cat",feature.shape)
        # print("voxelwise",voxelwise.shape)
     
     
        temp_hist_feature = feature[:,:,3:4]
        # print("voxel Intensity values shape",temp_hist_feature.shape)
        # hist_feature[:,:] = torch.histc(feature[:,:,1], bins=10, min=0, max=1) 
        
        # print("Printing hist feature dimension",voxelwise.shape[0])
        
        final_feature = torch.zeros((voxelwise.shape[0],10))
        for row in range(voxelwise.shape[0]):
            # print("element wise features",temp_hist_feature[row].shape)
            # print("input type", type(temp_hist_feature[row]))
            hist = torch.histc(temp_hist_feature[row], bins=10,min=0,max=1)
            # print(row,",hist values",hist)
            final_feature[row,:] = hist

        print("voxelwise",voxelwise.shape)
        print("final feature shape", final_feature.shape)
        # print("final feature",final_feature)
        
        if(voxelwise.is_cuda):
            print("in cuda")
        if(final_feature.is_cuda):
            print("in cude") 
        
        final_voxelwise = torch.cat((voxelwise,final_feature ),1)
        
        print("final_voxelwise", final_voxelwise.shape)

        # print("[ΣK, 138]",hist_feature.shape)

        # Car: [B, 10, 400, 352, 128]; Pedestrain/Cyclist: [B, 10, 200, 240, 128]
        # outputs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size([batch_size, cfg.INPUT_DEPTH, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))


        # Car: [B, 10, 400, 352, 128]; Pedestrain/Cyclist: [B, 10, 200, 240, 138]
        outputs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size([batch_size, cfg.INPUT_DEPTH, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 138]))



        outputs = outputs.to_dense()

        return outputs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def roi_align(feat, roi, pool_size):
	# INPUT:
	# feat: N * C * H * W
	# roi: N * roi_num_per_img * 4  x1 y1 x2 y2
	
	# Output:
	# crop_feat: N * roi_num_per_img * C * pool_size * pool_size

	N = feat.shape[0]
	C = feat.shape[1]
	H = feat.shape[2]
	W = feat.shape[3]
	roi_num_per_img = roi.shape[1]

	x1 = roi[...,0].view(-1)
	y1 = roi[...,1].view(-1)
	x2 = roi[...,2].view(-1)
	y2 = roi[...,3].view(-1)

	theta = Variable(roi.data.new(roi.size(0) * roi.size(1), 2, 3).zero_())
	theta[:, 0, 0] = (x2 - x1) / (W - 1)
	theta[:, 0 ,2] = (x1 + x2 - W + 1) / (W - 1)
	theta[:, 1, 1] = (y2 - y1) / (H - 1)
	theta[:, 1, 2] = (y1 + y2 - H + 1) / (H - 1)

	theta = theta.view(N, roi_num_per_img, 2, 3).view(N * roi_num_per_img, 2, 3)

	grid = F.affine_grid(theta, torch.Size((theta.size(0), 1, pool_size, pool_size)))
	feat = feat[:,None,...].expand(-1,roi_num_per_img,-1,-1,-1).contiguous().view(N * roi_num_per_img, C, H, W)
	crop_feat = F.grid_sample(feat, grid).view(N, roi_num_per_img, C, pool_size, pool_size)
	return crop_feat
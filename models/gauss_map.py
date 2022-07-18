import numpy as np
import torch
import math
import cv2
import os

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_proposal_pos_embed(proposals):
    num_pos_feats = 128
    temperature = 10000
    scale = 2 * math.pi

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    # N, L, 4
    proposals = proposals.sigmoid() * scale
    # N, L, 4, 128
    pos = proposals[:, :, :, None] / dim_t
    # N, L, 4, 64, 2
    pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
    return pos

def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return torch.from_numpy(np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type))

def check_point(rand_points, targets):
    for i in range(rand_points.shape[0]):
        count = 0
        cx = targets[i]['boxes'][0][0]
        cy = targets[i]['boxes'][0][1]
        w = targets[i]['boxes'][0][2]
        h = targets[i]['boxes'][0][3]
        for j in range(rand_points.shape[1]):
            if (rand_points[i][j][0] > (cx-w/2)) & (rand_points[i][j][0] < (cx+w/2)) & (rand_points[i][j][1] > (cy-h/2)) & (rand_points[i][j][1] < (cy+h/2)):
                count += 1
    
        if count==0:
            rand_points[i][0] = targets[i]['boxes'][0][:2]

    return rand_points

def init_points(bs, size):
    a = torch.tensor([20,40,60,80])
    x,y = torch.meshgrid(a,a)
    rand_points = torch.cat([x.flatten().unsqueeze(-1), y.flatten().unsqueeze(-1)], dim=-1).unsqueeze(0).repeat(bs, 1, 1)
    pos_emb = get_proposal_pos_embed(rand_points.float()/(size[0]-1)).permute(1, 0, 2)
    return pos_emb

def shuffle_tensor(input_tensor,dim=1):
    assert dim ==1 and input_tensor.dim()>=2
    idx = torch.randperm(input_tensor.shape[1])
    input_tensor = input_tensor[:, idx].view(input_tensor.size())
    return input_tensor

def shuffle_tensor_strong(input_tensor,dim=1):
    assert dim ==1 and input_tensor.dim()>=2
    for i in range(input_tensor.shape[0]):
        idx = torch.randperm(input_tensor.shape[1])
        input_tensor[i] = input_tensor[i, idx].view(input_tensor.size()[1:])
    return input_tensor

def make_all_points_gaussian(size, coord, score):

    device = coord.device
    coord = size[0] * coord.detach().cpu().numpy()
    score = normalization(score.detach().cpu().numpy())

    a = torch.tensor([20,40,60,80])
    x,y = torch.meshgrid(a,a)
    rand_points = torch.cat([x.flatten().unsqueeze(-1), y.flatten().unsqueeze(-1)], dim=-1).unsqueeze(0).repeat(coord.shape[0], 1, 1)

    final_coords = rand_points.float()
    pos_emb = get_proposal_pos_embed(final_coords/(size[0]-1)).permute(1, 0, 2).to(device)
    return final_coords, rand_points, pos_emb

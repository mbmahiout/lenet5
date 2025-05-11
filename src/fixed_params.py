import torch


def get_c3_connectivity():
    # See Table I in LeCun et al. 1998
    c3_connectivity = torch.zeros((6, 16))
    c3_connectivity[0, 0] = 1
    c3_connectivity[0, 4:7] = 1
    c3_connectivity[0, 9:13] = 1
    c3_connectivity[0, 14:16] = 1
    c3_connectivity[1, 0:2] = 1
    c3_connectivity[1, 5:8] = 1
    c3_connectivity[1, 10:14] = 1
    c3_connectivity[1, 15] = 1
    c3_connectivity[2, 0:3] = 1
    c3_connectivity[2, 6:9] = 1
    c3_connectivity[2, 11] = 1
    c3_connectivity[2, 13:16] = 1
    c3_connectivity[3, 1:4] = 1
    c3_connectivity[3, 6:10] = 1
    c3_connectivity[3, 12] = 1
    c3_connectivity[3, 14:16] = 1
    c3_connectivity[4, 2:5] = 1
    c3_connectivity[4, 7:11] = 1
    c3_connectivity[4, 12:14] = 1
    c3_connectivity[4, 15] = 1
    c3_connectivity[5, 3:6] = 1
    c3_connectivity[5, 8:12] = 1
    c3_connectivity[5, 13:16] = 1
    return c3_connectivity.t().float()

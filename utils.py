import torch
from torch.distributions import Normal, StudentT
import sys
import numpy as np

def wavelet_transform(x):
    '''
    :param x: Torch tensor, 2-dimensional time-series data (batch, window)
    :return: V1, W1 (scaled data and wavelet transformed data)
    '''

    if type(x).__module__ != torch.__name__:
        x = torch.from_numpy(x)
    batch_size = x.size(0)
    output_size = x.size(1) // 2
    s = 1/2  # scale

    # index for selecting odd, even column
    ids_odd = torch.ByteTensor([1, 0]).unsqueeze(0).repeat(batch_size, output_size)
    ids_even = torch.ByteTensor([0, 1]).unsqueeze(0).repeat(batch_size, output_size)

    # window size of data is odd,
    if x.size(1) % 2 != 0:
        output_size += 1
        ids_odd = torch.cat((ids_odd, torch.ByteTensor([1] * batch_size).view(-1, 1)), dim=1)
        ids_even = torch.cat((ids_even, torch.ByteTensor([1] * batch_size).view(-1, 1)), dim=1)

    cAs = s * (x[ids_odd] + x[ids_even]).view(batch_size, output_size)
    cDs = s * (x[ids_odd] - x[ids_even]).view(batch_size, output_size)

    return cAs, cDs

def get_log_prob(x, mean, logvar, dist='normal', v=3):
    if type(logvar) == float:
        mean = torch.ones_like(x)*mean
        std = torch.ones_like(x)*logvar
    else:
        std = logvar.mul(0.5).exp()
    if dist == 'normal':
        return Normal(mean, std).log_prob(x)
    elif dist == 't':
        return StudentT(v, mean, logvar).log_prob(x)
    else:
        print('there is no ', dist)
        sys.exit(0)

def match_date(basis_times, unmatched_table):
    '''
        matching dates copying past basis value
    :param basis_times: (np.ndarray)
    :param unmatched_table: (np.ndarray)
    :return:
    '''

    matched_shape = list(np.shape(unmatched_table))
    matched_shape[0] = np.shape(basis_times)[0]
    matched_table = np.empty(matched_shape, np.int64)

    assert basis_times[0] == unmatched_table[0,0]

    pvt = 0
    length = matched_shape[0]
    for idx in range(length):
        if basis_times[idx] != unmatched_table[pvt,0]:
            if basis_times[idx] > unmatched_table[pvt,0]:
                for i in range(pvt+1, np.shape(unmatched_table)[0], +1):
                    if basis_times[idx] >= unmatched_table[i,0]:
                        pvt = i
                    else:
                        break
            else: # basis < unmatched
                for i in range(pvt-1, -1, -1):
                    if basis_times[idx] <= unmatched_table[i,0]:
                        pvt = i
        matched_table[idx, :] = unmatched_table[pvt, :]
        pvt = min(pvt+1, length-1)

    return matched_table

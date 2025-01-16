# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
import torch


def ctc_reduce(hyp, blank_id: int = 0):
    new_hyp = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def ctc_peak_time(hyp, blank_id: int = 0):
    times = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            times.append(cur)
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return times


def lists_to_tensor(lists, padding_value=0, device='cpu'):
    max_len = max(len(l) for l in lists)
    padded_list = [l + [padding_value] * (max_len - len(l)) for l in lists]
    tensor = torch.tensor(padded_list, dtype=torch.long, device=device)
    return tensor

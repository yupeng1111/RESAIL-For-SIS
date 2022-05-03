from glob import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch


def match_score(
        class_name='bicycle',
        score_index=0,
        dir_p='/data/shiyupeng/SPADE/database/cityscapes/tensors/scores',
        topk=10,
        thresthold=2,
        device=1,
    ):

    size_score_file = 'size_score-' + class_name + '.pt'
    shape_score_file = 'shape_score-' + class_name + '.pt'

    size_score = torch.load(os.path.join(dir_p, size_score_file))
    shape_score = torch.load(os.path.join(dir_p, shape_score_file))
    size_score = torch.where(size_score < 1, torch.reciprocal(size_score), size_score)
    size_score = torch.where(size_score < thresthold, torch.zeros_like(size_score), torch.ones_like(size_score))
    score = shape_score + size_score
    score = score[score.shape[1]:, ...]
    if score.shape[0] == 0:
        return 0, 0

    try:
        topk = torch.topk(score, dim=1, k=topk + 1, largest=False)
        top_scores = topk[0]
        return torch.sum(top_scores[:, score_index]), top_scores[:, score_index].shape[0]
    except :
        return 0, 0


if __name__ == '__main__':
    dir_p = '/data/shiyupeng/SPADE/database/cityscapes/tensors/scores'
    topk = 1000

    names = [os.path.basename(p).split('.')[0].split('-')[1] for p in glob(dir_p + '/size_score-*.pt')]
    for score_index in [0,10,50,132,254,369,528,660,858,990]:
        sum_score = 0
        sum_n = 0
        for n in names:
            score, number = match_score(n, score_index=score_index, topk=topk)
            sum_score += score
            sum_n += number
        print(sum_score / sum_n)


import jieba
import editdistance
import numpy as np
# for simple demo
# def reward_len(completions, **kwargs):
#     return [-abs(20 - len(completion)) for completion in completions]

def editdistance_score(completions, **kwargs):
    # return [0] * len(completions)
    diff = []
    for hyp, lab in zip(completions,kwargs['txt']):
        # diff.append(-1.0 * editdistance.eval(hyp, lab) / len(lab))
        # diff.append(np.log(1e-9 + editdistance.eval(hyp, lab)))
        diff.append(-editdistance.eval(hyp, lab))
    return diff

def word_count(completions, **kwargs):
    c = []
    for i, completion in enumerate(completions):
        words = jieba.cut(completion.strip())
        words = [w.strip() for w in words]
        words = [w for w in words if w != '']
        if i==0:
            print(kwargs['wav'][i],words)
        c.append(-abs(0 - len(completion)))
    # print('-'*120)
    return c
    # return [len(list(jieba.cut(completion))) for completion in completions]

active_reward_func = [editdistance_score]
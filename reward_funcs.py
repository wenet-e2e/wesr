
import jieba

# for simple demo
# def reward_len(completions, **kwargs):
#     return [-abs(20 - len(completion)) for completion in completions]

def word_count(completions, **kwargs):
    c = []
    for completion in completions:
        words = jieba.cut(completion.strip())
        words = [w.strip() for w in words]
        words = [w for w in words if w != '']
        print(words)
        c.append(len(words))
    print('-'*120)
    return c
    # return [len(list(jieba.cut(completion))) for completion in completions]

active_reward_func = [word_count]
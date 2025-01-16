# Copyright (c) 2024 Binbin Zhang(binbzha@qq.com)
import sys
from dataclasses import dataclass, field

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer
from accelerate import Accelerator

from dataset import SpeechDataset, DataArguments
from speech_llm import init_model, ModelArguments


@dataclass
class DecodeArguments:
    result_path: str = field(default=None, metadata={"help": "Path to result"})


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DecodeArguments))
    model_args, data_args, decode_args = parser.parse_args_into_dataclasses()
    model = init_model(model_args)
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_name_or_path)
    test_dataset = SpeechDataset(data_args.data_path, tokenizer, model_args)
    data_loader = DataLoader(test_dataset, batch_size=1)
    if torch.cuda.is_available():
        model = model.cuda()
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    fid = open(decode_args.result_path, 'w', encoding='utf8')
    count = 0
    total = 0
    with torch.no_grad():
        for i, item in enumerate(tqdm(data_loader)):
            ss = model.compute_similarity(**item)
            for s in ss:
                slist = s.tolist()
                strs = ' '.join(['{:.6f}'.format(e) for e in slist])
                print(strs)
                fid.write(strs + '\n')
                total += sum(slist)
                count += len(slist)
            sys.stdout.flush()
            if i > 100:
                break
    fid.write('Average similarity {:.6f} Total token {}\n'.format(
        total / count, count))
    fid.close()


if __name__ == "__main__":
    main()

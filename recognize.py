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
    llm_type: str = 'qwen2'
    decode_type: str = 'llm'
    max_new_tokens: int = 50
    num_beams: int = 1
    batch_size: int = 1
    result_path: str = field(default=None, metadata={"help": "Path to result"})


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DecodeArguments))
    model_args, data_args, decode_args = parser.parse_args_into_dataclasses()
    model = init_model(model_args)
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_name_or_path)
    if decode_args.llm_type == 'qwen2':
        eos_token_id = tokenizer.convert_tokens_to_ids(
            ['<|endoftext|>', '<|im_end|>'])
    else:
        tokenizer.pad_token = '<|finetune_right_pad_id|>'
        eos_token_id = tokenizer.convert_tokens_to_ids(
            ['<|end_of_text|>', '<|eot_id|>'])
    print('eos_token_id', eos_token_id)
    test_dataset = SpeechDataset(data_args.data_path,
                                 tokenizer,
                                 model_args,
                                 inference=True)
    data_loader = DataLoader(test_dataset, batch_size=decode_args.batch_size)
    if torch.cuda.is_available():
        model = model.cuda()
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    fid = open(decode_args.result_path, 'w', encoding='utf8')
    if decode_args.decode_type == 'llm':
        decode_func = model.generate
    else:
        decode_func = model.decode_ctc
    with torch.no_grad():
        for item in tqdm(data_loader):
            generated_ids = decode_func(**item,
                                        eos_token_id=eos_token_id,
                                        decode_config=decode_args)
            text = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)
            print(text)
            for t in text:
                t = t.replace('\n', ' ')
                fid.write(t + '\n')
            sys.stdout.flush()
    fid.close()


if __name__ == "__main__":
    main()

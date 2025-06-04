# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math
import json
from dataclasses import dataclass, field
from typing import Dict

import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_pt_utils import LabelSmoother
import torch
import torch.nn.functional as F
import torchaudio
import transformers
import whisper


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    batch_size: int = field(default=1, metadata={"help": "batch size"})
    max_dynamic_size: int = field(
        default=0,
        metadata={
            "help":
            "max units for dynmamic batch size, it will override any"
            "value given in batch_size"
        })
    pack_size: int = field(
        default=0,
        metadata={
            "help":
            "size for sequence pack, it will override any value"
            "given in batch_size and max_dynamic_size"
        })
    max_speech_size: int = 1000


class SpeechDataset(IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        inference: bool = False,
    ):
        super(SpeechDataset, self).__init__()
        self.data_path = data_args.data_path
        self.tokenizer = tokenizer
        self.inference = inference
        if data_args.pack_size > 0:
            self.mode = 'pack'
            self.pack_size = data_args.pack_size
        elif data_args.max_dynamic_size > 0:
            self.mode = 'dynamic'
            self.max_dynamic_size = data_args.max_dynamic_size
        else:
            self.mode = 'static'
            self.batch_size = data_args.batch_size
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        except:
            self.world_size = 1
            self.rank = 0

    def _read_one(self):
        with open(self.data_path, "r") as f:
            for i, line in enumerate(f):
                if i % self.world_size == self.rank:
                    yield json.loads(line)

    def _extract(self, item):
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        audio, sample_rate = torchaudio.load(item['wav'])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        audio = audio * (1 << 15)
        # mel: (T, 80)
        mel = torchaudio.compliance.kaldi.fbank(audio,
                                                num_mel_bins=80,
                                                frame_length=25,
                                                frame_shift=10,
                                                dither=0.0,
                                                energy_floor=0.0,
                                                sample_frequency=16000)
        # TODO(Binbin Zhang): Refine to instruction + <AUDIO>
        ids_audio = [0] * (mel.size(0) // 8)  # 8 is the final subsampling rate
        tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
        instruction = 'Transcribe the speech'
        chat = [{"role": "user", "content": instruction}]
        content = item['txt']
        if self.inference:
            kwargs = {'add_generation_prompt': True}
        else:
            chat.append({"role": "assistant", "content": content})
            kwargs = {'add_generation_prompt': False}
        ids_text = self.tokenizer.apply_chat_template(chat,
                                                      tokenize=True,
                                                      **kwargs)
        if ids_text[0] != self.tokenizer.bos_token_id:
            ids_text = [self.tokenizer.bos_token_id] + ids_text
        if ids_text[-1] != self.tokenizer.eos_token_id and not self.inference:
            ids_text = ids_text + [self.tokenizer.eos_token_id]
        ids = ids_audio + ids_text
        tgt = tgt_audio + ids_text
        input_ids = torch.tensor(ids, dtype=torch.int)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'mel': mel,
        }

    def _pack_sequence(self, seqs):
        """
        Our base LLM will apply `shift_labels` on the labels. Assume we have:
        input_ids: <sos> a  b   c      <eos>  <sos> x y  z      <eos>
                   a     b  c   <eos>  N      x     y z  <eos>  N
                   N     a  b   c      <eos>  N     x y  z      <eos>
        The target should like above after `shift_labels`, where N is for
        ignore_index, we should ignore the target when the input_ids is <eos>

        """
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        input_ids = torch.tensor([0] * self.pack_size, dtype=torch.int)
        labels = torch.tensor([IGNORE_TOKEN_ID] * self.pack_size,
                              dtype=torch.long)
        position_ids = torch.tensor([0] * self.pack_size, dtype=torch.int)
        audio_offsets = torch.tensor([0] * len(seqs), dtype=torch.int)
        attention_mask = torch.zeros((self.pack_size, self.pack_size),
                                     dtype=torch.bool)
        audio_features = []
        offset = 0
        cu_seq_lens = [0]
        max_length = 0
        for i, seq in enumerate(seqs):
            audio_offsets[i] = offset
            seq_len = len(seq['input_ids'])
            input_ids[offset:offset + seq_len] = seq['input_ids']
            labels[offset] = IGNORE_TOKEN_ID
            labels[offset + 1:offset + seq_len] = seq['labels'][1:]
            cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
            max_length = max(max_length, seq_len)
            position_ids[offset:offset + seq_len] = torch.arange(
                seq_len, dtype=torch.int)
            audio_features.append(seq['mel'])
            offset += seq_len
        labels[offset] = IGNORE_TOKEN_ID
        audio_feature_lengths = torch.tensor(
            [t.size(0) for t in audio_features], dtype=torch.int)
        audio_features = pad_sequence(audio_features, batch_first=True)
        cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'audio_offsets': audio_offsets,
            'audio_features': audio_features,
            'audio_feature_lengths': audio_feature_lengths,
        }

    def _batch(self, seqs):
        audio_features = [s['mel'] for s in seqs]
        audio_feature_lengths = torch.tensor(
            [t.size(0) for t in audio_features], dtype=torch.int)
        audio_features = pad_sequence(audio_features, batch_first=True)
        input_ids = pad_sequence([s['input_ids'] for s in seqs],
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([s['labels'] for s in seqs],
                              batch_first=True,
                              padding_value=LabelSmoother.ignore_index)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'audio_features': audio_features,
            'audio_feature_lengths': audio_feature_lengths,
        }

    def __iter__(self) -> Dict[str, torch.Tensor]:
        buffer = []
        total_length = 0
        for item in self._read_one():
            data = self._extract(item)
            if data['mel'].size(0) > 1000 and not self.inference:
                continue
            if self.mode == 'static' and len(buffer) == self.batch_size:
                yield self._batch(buffer)
                buffer = []
                total_length = 0
            elif self.mode == 'dynamic' and total_length + len(
                    data['input_ids']) > self.max_dynamic_size:
                yield self._batch(buffer)
                buffer = []
                total_length = 0
            elif self.mode == 'pack' and total_length + len(
                    data['input_ids']) >= self.pack_size:
                yield self._pack_sequence(buffer)
                buffer = []
                total_length = 0
            buffer.append(data)
            total_length += len(data['input_ids'])
        if self.mode in ['static', 'dynamic']:
            yield self._batch(buffer)
        else:
            yield self._pack_sequence(buffer)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import numpy as np
    tokenizer = AutoTokenizer.from_pretrained(
        '/bucket/output/jfs-hdfs/user/binbin.zhang/huggingface/hub/Qwen2-1.5B-Instruct'
    )
    tokenizer.bos_token = tokenizer.eos_token
    print(tokenizer.bos_token_id)
    data_args = DataArguments
    data_args.data_path = 'data/aishell/train.jsonl'
    data_args.pack_size = 256
    dataset = SpeechDataset(tokenizer, data_args)
    for i, x in enumerate(dataset):
        # np.savetxt('tensor.txt', x['attention_mask'].numpy(), fmt='%.4f')
        print(x)
        if i > 0: break

# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math
import json
from dataclasses import dataclass, field
from typing import Dict

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
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."})
    test_data_path: str = field(default=None,
                                metadata={"help": "Path to the test data."})


class SpeechDataset(IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer: transformers.PreTrainedTokenizer,
        inference: bool = False,
    ):
        super(SpeechDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.inference = inference
        self.max_pack_length = 8192
        self.mode = 'pack'  # static/dynamic/pack
        self.static_batch_size = 32
        self.max_dynamic_length = 4096

    def _read_one(self):
        with open(self.data_path, "r") as f:
            for line in f:
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
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        input_ids = torch.tensor([0] * self.max_pack_length, dtype=torch.int)
        labels = torch.tensor([IGNORE_TOKEN_ID] * self.max_pack_length,
                              dtype=torch.long)
        position_ids = torch.tensor([0] * self.max_pack_length, dtype=torch.int)
        audio_offsets = torch.tensor([0] * len(seqs), dtype=torch.int)
        audio_features = []
        offset = 0
        for i, seq in enumerate(seqs):
            audio_offsets[i] = offset
            seq_len = len(seq['input_ids'])
            input_ids[offset:offset + seq_len] = seq['input_ids']
            labels[offset:offset + seq_len] = seq['labels']
            position_ids[offset:offset + seq_len] = torch.arange(
                seq_len, dtype=torch.int)
            audio_features.append(seq['mel'])
            offset += seq_len
        audio_feature_lengths = torch.tensor(
            [t.size(0) for t in audio_features], dtype=torch.int)
        audio_features = pad_sequence(audio_features, batch_first=True)
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
            if self.mode == 'static' and len(buffer) == self.static_batch_size:
                yield self._batch(buffer)
                buffer = []
                total_length = 0
            elif self.mode == 'dynamic' and total_length + len(
                    data['input_ids']) > self.max_dynamic_length:
                yield self._batch(buffer)
                buffer = []
                total_length = 0
            elif self.mode == 'pack' and total_length + len(
                    data['input_ids']) > self.max_pack_length:
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
    tokenizer = AutoTokenizer.from_pretrained(
        '/bucket/output/jfs-hdfs/user/binbin.zhang/huggingface/hub/Qwen2-1.5B-Instruct'
    )
    dataset = SpeechDataset('data/aishell/train.jsonl', tokenizer)
    for i, x in enumerate(dataset):
        print(x)
        if i >= 1:
            break

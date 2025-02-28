# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math
import json
from dataclasses import dataclass, field
from typing import Dict

from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
import torch
import torchaudio
import transformers
import whisper
from torch.nn.utils.rnn import pad_sequence


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."})
    test_data_path: str = field(default=None,
                                metadata={"help": "Path to the test data."})
    max_tokens_in_batch: int = field(
        default=2000,
        metadata={"help": "the maximum number of tokens in a batch"})
    batch_type: str = field(default="static",
                            metadata={"help": "static or dynamic"})
    batch_size: int = field(
        default=8, metadata={"help": "number of utterances in a batch"})
    sort: bool = field(
        default=False,
        metadata={
            "help":
            "whether to sort all data, so the utterance with the same length could be filled in a same batch"
        })
    text_token_per_second: int = field(
        default=8, metadata={"help": "number of text tokens per second"})


@dataclass
class CustomDataCollator:
    ds_rate: int = 8
    pad_token_id: int = -1
    ignore_token_id: int = LabelSmoother.ignore_index

    def _padding(self, batch):
        # padding feats
        feats = [x['mel'] for x in batch]
        feats_length = torch.tensor([x['mel'].shape[0] for x in batch],
                                    dtype=torch.int64)
        padded_feats = pad_sequence(feats, batch_first=True,
                                    padding_value=0).transpose(1, 2)  # [80, T]
        max_speech_token_size = math.ceil(
            torch.max(feats_length) / self.ds_rate)

        # padding tokens
        ids_audio = torch.cat([
            torch.tensor([0] * max_speech_token_size).unsqueeze(0)
            for _ in batch
        ],
                              dim=0)
        tgt_audio = torch.cat([
            torch.tensor(
                [self.ignore_token_id] * max_speech_token_size).unsqueeze(0)
            for _ in batch
        ],
                              dim=0)

        ids_text = [x['label_ids'] for x in batch]
        padded_ids_text = pad_sequence(ids_text,
                                       batch_first=True,
                                       padding_value=self.pad_token_id)
        input_ids = torch.cat([ids_audio, padded_ids_text], dim=1)
        target_ids = torch.cat([tgt_audio, padded_ids_text], dim=1)

        target_ids[target_ids == self.pad_token_id] = self.ignore_token_id
        attention_mask = input_ids.ne(self.pad_token_id)

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mel": padded_feats,
            "mel_len": feats_length,
        }
        if 'ctc_ids' in batch[0]:
            ctc_ids = [x['ctc_ids'] for x in batch]
            padded_ctc_ids = pad_sequence(ctc_ids,
                                          batch_first=True,
                                          padding_value=self.pad_token_id)
            ctc_ids_len = torch.cat(
                [x['ctc_ids_len'].unsqueeze(0) for x in batch], dim=0)
            ret['ctc_ids'] = padded_ctc_ids
            ret['ctc_ids_len'] = ctc_ids_len
            ret['labels'] = target_ids
        return ret

    def __call__(self, batch):
        assert len(batch) == 1
        return self._padding(batch[0])


class SpeechDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer: transformers.PreTrainedTokenizer,
        config,  # model config
        inference: bool = False,
        batch_type: str = 'static',
        batch_size: int = 8,
        max_tokens_in_batch: int = 2000,
        sort: bool = False,
        text_token_per_second: int = 8,
    ):
        super(SpeechDataset, self).__init__()
        print("Formatting inputs...")
        self.tokenizer = tokenizer
        self.config = config
        self.inference = inference
        self.raw_data = []
        with open(data_path, "r") as f:
            for line in f:
                self.raw_data.append(json.loads(line))

        if sort:
            self.raw_data = sorted(self.raw_data, key=lambda x: x['duration'])

        self.minibatch = []
        num_data = len(self.raw_data)
        if batch_type == "dynamic":
            assert max_tokens_in_batch > 0
            self.minibatch.append([])
            num_tokens_in_batch = 0
            for i in range(num_data):
                length = self.raw_data[i][
                    'duration'] * self.config.speech_tokens_per_second
                if 'label_ids' in self.raw_data[i]:
                    length += len(self.raw_data[i]['label_ids'])
                else:
                    length += int(text_token_per_second *
                                  self.raw_data[i]['duration'])
                num_tokens_in_batch += length
                if num_tokens_in_batch > max_tokens_in_batch:
                    self.minibatch.append([])
                    num_tokens_in_batch = length
                self.minibatch[-1].append(self.raw_data[i])
        else:
            cur = 0
            while cur < num_data:
                self.minibatch.append(self.raw_data[cur:cur + batch_size])
                cur += batch_size

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self._extract_feature_and_token(self.minibatch[i])

    def _extract_feature_and_token(self, batch):
        processed_batch = []
        for msg in batch:
            audio, sample_rate = torchaudio.load(msg['wav'])
            if sample_rate != 16000:
                audio = torchaudio.functional.resample(audio, sample_rate,
                                                       16000)
            if self.config.encoder_type == 'whisper':
                mel_len = math.ceil(
                    float(audio.size(1)) / 16000 *
                    self.config.frames_per_second)
                audio = whisper.pad_or_trim(audio[0])
                mel = whisper.log_mel_spectrogram(audio).transpose(0,
                                                                   1)  # [T, 80]
            else:
                audio = audio * (1 << 15)
                mel = torchaudio.compliance.kaldi.fbank(audio,
                                                        num_mel_bins=80,
                                                        frame_length=25,
                                                        frame_shift=10,
                                                        dither=0.0,
                                                        energy_floor=0.0,
                                                        sample_frequency=16000)
                mel_len = mel.size(0)
            if 'instruction' in msg:
                instruction = msg['instruction']
            elif self.inference and self.config.decode_instruction != '':
                instruction = self.config.decode_instruction
            else:
                instruction = "Transcribe the speech"
            chat = [{"role": "user", "content": instruction}]
            content = msg['content'] if 'content' in msg else msg['txt']
            if self.inference:
                kwargs = {'add_generation_prompt': True}
            else:
                chat.append({"role": "assistant", "content": content})
                kwargs = {'add_generation_prompt': False}

            ids_text = self.tokenizer.apply_chat_template(chat,
                                                          tokenize=True,
                                                          **kwargs)
            ids_text = torch.tensor(ids_text, dtype=torch.int)

            ctc_tokens = self.tokenizer(msg['txt'], return_tensors='pt')
            ctc_ids = ctc_tokens['input_ids'][0]
            ctc_ids_len = torch.tensor(
                ctc_tokens['attention_mask'].sum().item(), dtype=torch.int)

            processed_batch.append({
                'label_ids': ids_text,
                'mel': mel,
                'mel_len': mel_len,
            })
            if not self.inference:
                processed_batch[-1]['ctc_ids'] = ctc_ids
                processed_batch[-1]['ctc_ids_len'] = ctc_ids_len
        return processed_batch

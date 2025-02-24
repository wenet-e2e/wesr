# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math
import random
import json
from dataclasses import dataclass, field
from typing import Dict

from torch.utils.data import Dataset, Sampler
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


@dataclass
class CustomDataCollator:
    ds_rate: int = 8
    pad_token_id: int = -1
    ignore_token_id: int = LabelSmoother.ignore_index

    def __call__(self, batch):
        """
        [{"mel", "mel_len", "label_ids", "ctc_ids", "ctc_ids_len"},...]

        """
        assert isinstance(batch, list)
        feats = [x['mel'] for x in batch]
        feats_length = torch.tensor([x['mel'].shape[0] for x in batch],
                                    dtype=torch.int64)
        max_feats_length = torch.max(feats_length)

        padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
        padded_feats = padded_feats.transpose(1, 2)  # [80, T]
        max_speech_token_size = math.ceil(max_feats_length / self.ds_rate)

        ids_audio = torch.cat([
            torch.tensor([0] * max_speech_token_size).unsqueeze(0)
            for _ in batch], dim=0)
        tgt_audio = torch.cat([
            torch.tensor(
                [self.ignore_token_id] * max_speech_token_size).unsqueeze(0)
            for _ in batch], dim=0)

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


class DynamicBatchSampler(Sampler):

    def __init__(self, dataset, max_tokens_in_batch, ds_rate):
        self.dataset = dataset
        self.max_tokens_in_batch = max_tokens_in_batch
        self.ds_rate = ds_rate
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)
        self._buffer = []
        self.longest_ids_length = 0
        self.longest_speech_token = 0

    def dynamic_batch_window(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'mel' in sample
        assert 'label_ids' in sample
        new_speech_token = math.ceil(sample['mel'].size(1) / self.ds_rate)
        self.longest_speech_token = max(self.longest_speech_token,
                                        new_speech_token)
        new_ids_length = sample['label_ids'].size(0)
        self.longest_ids_length = max(self.longest_ids_length, new_ids_length)

        tokens_after_padding = (self.longest_speech_token +
                                self.longest_ids_length) * (buffer_size + 1)
        if tokens_after_padding > self.max_tokens_in_batch:
            self.longest_speech_token = new_speech_token
            self.longest_ids_length = new_ids_length
            return True
        return False

    def __iter__(self):
        for idx in self.indices:
            if not self.dynamic_batch_window(self.dataset[idx], len(
                    self._buffer)):
                self._buffer.append(idx)
            else:
                if len(self._buffer) > 0:
                    yield self._buffer
                del self._buffer
                self._buffer = [idx]
        if len(self._buffer) > 0:
            yield self._buffer
        del self._buffer
        self._buffer = []

    def __len__(self):
        return len(self.indices)


class SpeechDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer: transformers.PreTrainedTokenizer,
        config,  # model config
        inference: bool = False,
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

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        msg = self.raw_data[i]
        audio, sample_rate = torchaudio.load(msg['wav'])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        if self.config.encoder_type == 'whisper':
            mel_len = math.ceil(
                float(audio.size(1)) / 16000 * self.config.frames_per_second)
            audio = whisper.pad_or_trim(audio[0])
            mel = whisper.log_mel_spectrogram(audio)  # [80, T]
            mel = mel.transpose(0, 1)  # [T, 80]
        else:
            # Note: We use 16-bit quantization by default in WeNet.
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
            instruction = 'Transcribe the speech'
        chat = [{"role": "user", "content": instruction}]
        # `content`: the anwser acorrding to the audio and instruction
        # `txt`: the transcription of the audio
        # If there is no content, the default `content` is the same as `txt`.
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
        ctc_ids_len = torch.tensor(ctc_tokens['attention_mask'].sum().item(),
                                   dtype=torch.int)
        ret = {
            'label_ids': ids_text,
            'mel': mel,
            'mel_len': mel_len,
        }
        if not self.inference:
            ret['ctc_ids'] = ctc_ids
            ret['ctc_ids_len'] = ctc_ids_len
        return ret

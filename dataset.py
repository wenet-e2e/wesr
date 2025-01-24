# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math
import json
from dataclasses import dataclass, field
from typing import Dict

from torch.utils.data import Dataset
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
            mel = whisper.log_mel_spectrogram(audio)
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
            mel = mel.transpose(0, 1)  # (80, T)
            if mel.size(1) < self.config.max_mel_size:
                mel_len = mel.size(1)
                mel = F.pad(mel, (0, self.config.max_mel_size - mel.size(1)),
                            value=0.0)
            else:  # hard truncation
                mel_len = self.config.max_mel_size
                mel = mel[:, :self.config.max_mel_size]
        ids_audio = [0] * self.config.max_speech_token_size
        tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
        chat = [{"role": "user", "content": "Transcribe the speech"}]
        if self.inference:
            kwargs = {'add_generation_prompt': True}
        else:
            chat.append({"role": "assistant", "content": msg['txt']})
            kwargs = {
                'padding': 'max_length',
                'max_length': self.config.model_max_length -
                self.config.max_speech_token_size,
                'truncation': True,
                'add_generation_prompt': False,
            }
        ids_text = self.tokenizer.apply_chat_template(chat,
                                                      tokenize=True,
                                                      **kwargs)
        ids = ids_audio + ids_text
        tgt = tgt_audio + ids_text
        input_ids = torch.tensor(ids, dtype=torch.int)
        target_ids = torch.tensor(tgt, dtype=torch.int)
        target_ids[target_ids == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        ctc_tokens = self.tokenizer(msg['txt'],
                                    padding='max_length',
                                    max_length=100,
                                    truncation=True,
                                    return_tensors='pt')
        ctc_ids = ctc_tokens['input_ids'][0]
        ctc_ids_len = ctc_tokens['attention_mask'].sum().item()
        ret = {
            'input_ids': input_ids,
            'labels': target_ids,
            'attention_mask': attention_mask,
            'mel': mel,
            'mel_len': mel_len,
        }
        if not self.inference:
            ret['ctc_ids'] = ctc_ids
            ret['ctc_ids_len'] = ctc_ids_len
        return ret

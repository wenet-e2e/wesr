# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import json
from dataclasses import dataclass, field
from typing import Dict

from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
import torch
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
        max_len: int = 512,
        inference: bool = False,
        n_mels: int = 80,
    ):
        super(SpeechDataset, self).__init__()
        print("Formatting inputs...")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inference = inference
        self.n_mels = n_mels
        self.raw_data = []
        with open(data_path, "r") as f:
            for line in f:
                self.raw_data.append(json.loads(line))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        msg = self.raw_data[i]
        # load audio and pad/trim it to fit 30 seconds
        speech_len = 300
        audio, sample_rate = torchaudio.load(msg['wav'])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        audio = audio[0]  # get the first channel
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        ids_audio = [0] * int(mel.shape[1] / 10)  # 10x downsample
        tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
        chat = [{"role": "user", "content": "Transcribe the speech"}]
        if self.inference:
            kwargs = {'add_generation_prompt': True}
        else:
            chat.append({"role": "assistant", "content": msg['txt']})
            kwargs = {
                'padding': 'max_length',
                'max_length': self.max_len - speech_len,
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
        return {
            'input_ids': input_ids,
            'labels': target_ids,
            'attention_mask': attention_mask,
            'mel': mel,
        }

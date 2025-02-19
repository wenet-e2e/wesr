# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import torch
import math

from functools import partial
import sys
from typing import Optional
from wenet.dataset import processor
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_pt_utils import LabelSmoother


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000, ds_rate=8):
        self.longest_frames = 0
        self.longest_mel = 0
        self.longest_text = 0
        self.max_frames_in_batch = max_frames_in_batch
        self.ds_rate = ds_rate

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        self.longest_mel = max(self.longest_mel, new_sample_frames)
        ids_text_length = len(sample['ids_text'])
        self.longest_text = max(self.longest_text, ids_text_length)
        self.longest_frames = max(
            self.longest_frames,
            self.longest_mel // self.ds_rate + self.longest_text)

        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames // self.ds_rate + ids_text_length
            self.longest_mel = new_sample_frames
            self.longest_text = ids_text_length
            return True
        return False


def tokenize(sample, tokenizer, inference=False, decode_instruction=""):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    assert 'txt' in sample
    if "instruction" in sample:
        instruction = sample['instruction']
    elif inference and decode_instruction != '':
        instruction = self.config.decode_instruction
    else:
        instruction = 'Transcribe the speech'
    chat = [{"role": "user", "content": instruction}]
    content = sample['content'] if 'content' in sample else sample['txt']
    if inference:
        kwargs = {'add_generation_prompt': True}
    else:
        chat.append({"role": "assistant", "content": content})
        kwargs = {
            'add_generation_prompt': False,
        }
    ids_text = tokenizer.apply_chat_template(chat, tokenize=True, **kwargs)
    sample["ids_text"] = ids_text

    return sample


def gather(samples,
           tokenizer,
           max_ids_text_size=512,
           max_speech_token_size=300,
           inference=False,
           decode_instruction=""):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    IGNORE_TOKEN_ID = LabelSmoother.ignore_index

    input_ids_list = []
    target_ids_list = []
    attention_mask_list = []
    ctc_ids_list = []
    ctc_ids_len_list = []

    for sample in samples:
        ids_audio = [0] * max_speech_token_size
        tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
        if "instruction" in sample:
            instruction = sample['instruction']
            assert False
        elif inference and decode_instruction != '':
            instruction = self.config.decode_instruction
            assert False
        else:
            instruction = 'Transcribe the speech'
        chat = [{"role": "user", "content": instruction}]
        content = sample['content'] if 'content' in sample else sample['txt']
        if inference:
            kwargs = {'add_generation_prompt': True}
            assert False
        else:
            chat.append({"role": "assistant", "content": content})
            kwargs = {
                'padding': 'max_length',
                'max_length': max_ids_text_size,
                'truncation': True,
                'add_generation_prompt': False,
            }
        ids_text = tokenizer.apply_chat_template(chat, tokenize=True, **kwargs)
        ids = ids_audio + ids_text
        tgt = tgt_audio + ids_text

        input_ids = torch.tensor(ids, dtype=torch.int)
        target_ids = torch.tensor(tgt, dtype=torch.int)
        target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        ctc_tokens = tokenizer(sample['txt'],
                               padding="max_length",
                               max_length=max_ids_text_size,
                               truncation=True,
                               return_tensors="pt")
        ctc_ids = ctc_tokens["input_ids"][0]
        ctc_ids_len = torch.tensor(ctc_tokens['attention_mask'].sum().item(),
                                   dtype=torch.int64)
        input_ids_list.append(input_ids)
        target_ids_list.append(target_ids)
        attention_mask_list.append(attention_mask)
        ctc_ids_list.append(ctc_ids)
        ctc_ids_len_list.append(ctc_ids_len)

    return input_ids_list, target_ids_list, attention_mask_list, ctc_ids_list, ctc_ids_len_list


def padding(data, tokenizer, model_args):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)

    feats = [x['feat'] for x in sample]
    # feats_length = torch.tensor([x['mel_len'] for x in sample], dtype=torch.int64)
    feats_length = torch.tensor([x['feat'].shape[0] for x in sample],
                                dtype=torch.int64)
    max_feats_length = torch.max(feats_length)

    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    padded_feats = padded_feats.transpose(1, 2)  # [80, T]

    max_speech_size = math.ceil(max_feats_length / model_args.ds_rate)
    # max labels
    ids_text_lengths = torch.tensor([len(x['ids_text']) for x in sample],
                                    dtype=torch.int32)
    max_ids_text_lengths = torch.max(ids_text_lengths)

    input_ids, labels, attention_mask, ctc_ids, ctc_ids_len = gather(
        sample,
        tokenizer,
        max_ids_text_size=max_ids_text_lengths,
        max_speech_token_size=max_speech_size,
        inference=False)

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "mel": padded_feats,
        "mel_len": feats_length,  # feats_length
        "ctc_ids": ctc_ids,
        "ctc_ids_len": ctc_ids_len,
        # "max_feats_length": max_feats_length,
        # "max_ids_text_size": max_ids_text_lengths,
        # "feats_lengths": feats_lengths,
    }

    return batch


def Dataset(data_type,
            data_list_file,
            tokenizer=None,
            conf=None,
            partition=True,
            model_args=None,
            inference=False):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert conf is not None
    assert data_type in ['raw', 'shard']
    assert tokenizer is not None
    # cycle dataset
    cycle = conf.get('cycle', 1)
    # stage1 shuffle: source
    list_shuffle = conf.get('list_shuffle', True)
    list_shuffle_size = sys.maxsize
    if list_shuffle:
        list_shuffle_conf = conf.get('list_shuffle_conf', {})
        list_shuffle_size = list_shuffle_conf.get('shuffle_size',
                                                  list_shuffle_size)
    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file,
                                        partition=partition,
                                        shuffle=list_shuffle,
                                        shuffle_size=list_shuffle_size,
                                        cycle=cycle)
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file,
                                             partition=partition,
                                             shuffle=list_shuffle,
                                             shuffle_size=list_shuffle_size,
                                             cycle=cycle)
    dataset = dataset.map_ignore_error(processor.decode_wav)

    singal_channel_conf = conf.get('singal_channel_conf', {})
    dataset = dataset.map(
        partial(processor.singal_channel, **singal_channel_conf))

    dataset = dataset.map(
        partial(tokenize,
                tokenizer=tokenizer,
                inference=inference,
                decode_instruction=""))

    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(processor.filter, **filter_conf))

    resample_conf = conf.get('resample_conf', {})
    dataset = dataset.map(partial(processor.resample, **resample_conf))

    feats_type = conf.get('feats_type', 'fbank_pad')
    assert feats_type in ['log_mel_spectrogram', "fbank", "fbank_pad"]
    if feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = dataset.map(
            partial(processor.compute_whisper_log_mel_spectrogram,
                    **log_mel_spectrogram_conf))
    elif feats_type == "fbank":
        fbank_conf = conf.get('fbank_conf', {})
        dataset = dataset.map(partial(processor.compute_fbank, **fbank_conf))
    elif feats_type == "fbank_pad":
        fbank_pad_conf = conf.get("fbank_pad_conf", {})
        fbank_pad_conf["max_mel_size"] = model_args.max_mel_size
        print("max_mel_size: ", model_args.max_mel_size)
        dataset = dataset.map(
            partial(processor.compute_fbank_pad, **fbank_pad_conf))

    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    assert batch_type in ['static', 'dynamic']
    batch_mode = batch_conf.get('batch_mode', 'asr')
    assert batch_mode in ['asr', 'tts', 'asr_tts']
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(batch_size,
                                wrapper_class=partial(padding,
                                                      tokenizer=tokenizer,
                                                      model_args=model_args))

    else:
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        print("dynamic batch, max_frames_in_batch: ", max_frames_in_batch)
        dataset = dataset.dynamic_batch(
            DynamicBatchWindow(max_frames_in_batch, model_args.ds_rate),
            wrapper_class=partial(padding,
                                  tokenizer=tokenizer,
                                  model_args=model_args),
        )

    return dataset


def init_dataset(train_data,
                 data_type,
                 configs_file,
                 tokenizer=None,
                 model_args=None):
    assert configs_file is not None
    with open(configs_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    train_conf = configs['dataset_conf']
    train_dataset = Dataset(data_type, train_data, tokenizer, train_conf, True,
                            model_args)
    return train_dataset

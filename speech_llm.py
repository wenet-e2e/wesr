# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import logging
import math
from typing import Optional
from dataclasses import dataclass, field

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel
import wenet
import whisper


@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    whisper_model_name_or_path: Optional[str] = field(default="tiny")
    wenet_model_name_or_path: Optional[str] = field(default="")
    encoder_type: str = field(
        default="whisper",
        metadata={"help": "encoder type, whisper or wenet"},
    )
    encoder_ds_rate: int = 2
    encoder_projector_ds_rate: int = 5
    projector_hidden_size: int = 2048
    projector_model_path: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"},
    )
    max_speech_seconds: int = 30
    frames_per_second: int = 100
    # For decode
    decode_instruction: Optional[str] = field(default="")

    @property
    def ds_rate(self):
        return self.encoder_ds_rate * self.encoder_projector_ds_rate

    @property
    def speech_tokens_per_second(self):
        return self.frames_per_second / self.ds_rate

    @property
    def max_speech_token_size(self):
        return math.ceil(self.max_speech_seconds *
                         self.speech_tokens_per_second)

    @property
    def max_mel_size(self):
        return self.max_speech_seconds * self.frames_per_second


class ProjectorCov1d(nn.Module):

    def __init__(self, config, encoder_dim, llm_dim):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.conv1d = nn.Conv1d(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=self.k,
                                stride=self.k,
                                padding=0)
        self.linear1 = nn.Linear(encoder_dim, config.projector_hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(config.projector_hidden_size, llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class SpeechLLM(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        llm: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
        config,
        model_args: ModelArguments,
    ):
        super().__init__(config)
        self.llm = llm
        self.encoder = encoder
        self.projector = projector
        self._keys_to_ignore_on_save = set()
        # Do not save the parameter of llm and whisper
        for k in self.llm.state_dict().keys():
            self._keys_to_ignore_on_save.add('llm.' + k)
        for k in self.encoder.state_dict().keys():
            self._keys_to_ignore_on_save.add('encoder.' + k)
        self.model_args = model_args
        self.num_setences = 0

    def get_speech_embeddings(self, audio_features, audio_feature_lengths):
        speech_emb, mask = self.encoder._forward_encoder(
            audio_features, audio_feature_lengths)
        speech_emb = speech_emb.masked_fill(~mask.transpose(1, 2), 0.0)
        speech_proj = self.projector(speech_emb)
        speech_proj_lens = mask.squeeze(1).sum(1) // self.projector.k
        return speech_proj, speech_proj_lens

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb, speech_emb_lens = self.get_speech_embeddings(
            audio_features, audio_feature_lengths)
        inputs_embeds = text_emb
        if audio_offsets is None:  # batch
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                inputs_embeds[i, :speech_emb_lens[i], :] = speech_emb[
                    i, :speech_emb_lens[i], :]
            out = self.llm(inputs_embeds=inputs_embeds,
                           attention_mask=attention_mask,
                           labels=labels)
            self.num_setences += batch_size
        else:
            batch_size = audio_offsets.size(0)
            for i in range(batch_size):
                s, e = audio_offsets[i], audio_offsets[i] + speech_emb_lens[i]
                inputs_embeds[s:e, :] = speech_emb[i, :speech_emb_lens[i], :]
            out = self.llm(inputs_embeds=inputs_embeds.unsqueeze(0),
                           position_ids=position_ids.unsqueeze(0),
                           labels=labels.unsqueeze(0))
            self.num_setences += batch_size
        logging.info('Train finish {} sentences'.format(self.num_setences))
        return out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        mel: torch.LongTensor = None,
        mel_len: torch.LongTensor = None,
        eos_token_id=None,
        decode_config=None,
    ):
        max_speech_size = self.model_args.max_speech_token_size
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb = self.get_speech_embeddings(mel, mel_len)
        inputs_embeds = torch.cat(
            (speech_emb, text_emb[:, max_speech_size:, :]), dim=1)
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=1.0,
            num_beams=decode_config.num_beams,
            max_new_tokens=decode_config.max_new_tokens,
            eos_token_id=eos_token_id,
        )
        return model_outputs

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_encoder(self):
        freeze_model(self.encoder)
        self.encoder.eval()

    def freeze_llm(self):
        freeze_model(self.llm)

    def load_projector(self, projector_path):
        projector_state_dict = safetensors.torch.load_file(projector_path)
        self.load_state_dict(projector_state_dict, strict=False)


def init_model(model_args):
    if model_args.encoder_type == "whisper":
        encoder = whisper.load_model(model_args.whisper_model_name_or_path)
    elif model_args.encoder_type == "wenet":
        encoder = wenet.load_model_pt(model_args.wenet_model_name_or_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = encoder.to(device)
    else:
        raise ValueError(f"Unexpected encoder type {model_args.encoder_type}")

    # Load llm model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.llm_model_name_or_path)
    config.use_cache = False
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        config=config,
        torch_dtype='auto',
        attn_implementation="flash_attention_2",
    )
    if model_args.encoder_type == "whisper":
        encoder_dim = encoder.dims.n_audio_state
    else:
        encoder_dim = encoder.encoder.output_size()
    llm_dim = config.hidden_size
    projector = ProjectorCov1d(model_args, encoder_dim, llm_dim)
    total_params = sum(p.numel() for p in projector.parameters())
    print('Projector total params: {:.2f}M'.format(total_params / 1024 / 1024))
    model = SpeechLLM(llm_model, encoder, projector, config, model_args)
    if model_args.projector_model_path is not None:
        model.load_projector(model_args.projector_model_path)
    return model

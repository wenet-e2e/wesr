# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Optional
from dataclasses import dataclass, field

import safetensors
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel
import whisper


@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    whisper_model_name_or_path: Optional[str] = field(default="tiny")
    encoder_ds_rate: int = 2
    encoder_projector_ds_rate: int = 5
    projector_hidden_size: int = 2048
    projector_model_path: Optional[str] = field(default=None)


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
        config,
        llm: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
    ):
        super().__init__(config)
        self.config = config  # copy llm's config
        self.llm = llm
        self.encoder = encoder
        self.projector = projector
        self._keys_to_ignore_on_save = set()
        # Do not save the parameter of llm and whisper
        for k in self.llm.state_dict().keys():
            self._keys_to_ignore_on_save.add('llm.' + k)
        for k in self.encoder.state_dict().keys():
            self._keys_to_ignore_on_save.add('encoder.' + k)

    def get_input_embedding(self, input_ids, mel):
        # whisper, 30s, 2x downsample = 1500
        speech_size = 300
        speech_emb = self.encoder.embed_audio(mel)  # (b, n_mel, 1500)
        # projector, x 5x downsample = 300
        speech_proj = self.projector(speech_emb)  # (b, x, 300)
        text_emb = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat((speech_proj, text_emb[:, speech_size:, :]),
                                  dim=1)
        return inputs_embeds

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mel: torch.LongTensor = None,
    ):
        inputs_embeds = self.get_input_embedding(input_ids, mel)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mel: torch.LongTensor = None,
        eos_token_id=None,
        decode_config=None,
    ):
        inputs_embeds = self.get_input_embedding(input_ids, mel)
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

    def freeze_llm(self):
        freeze_model(self.llm)

    def load_projector(self, projector_path):
        projector_state_dict = safetensors.torch.load_file(projector_path)
        self.load_state_dict(projector_state_dict, strict=False)


def init_model(model_args):
    encoder = whisper.load_model(model_args.whisper_model_name_or_path)
    # Load llm model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.llm_model_name_or_path)
    config.use_cache = False
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        config=config,
        torch_dtype='auto',
    )
    encoder_dim = encoder.dims.n_audio_state
    llm_dim = config.hidden_size
    projector = ProjectorCov1d(model_args, encoder_dim, llm_dim)
    total_params = sum(p.numel() for p in projector.parameters())
    print('Projector total params: {:.2f}M'.format(total_params / 1024 / 1024))
    model = SpeechLLM(config, llm_model, encoder, projector)
    if model_args.projector_model_path is not None:
        model.load_projector(model_args.projector_model_path)
    return model, encoder.dims.n_mels

# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math
from typing import Optional
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional, Sized, Union

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
import wenet
import whisper
from whisper.model import ModelDimensions, Whisper
import inspect


@dataclass
class ModelArguments:

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield (field.name, getattr(self, field.name))
        for attr, value in inspect.getmembers(self.__class__):
            if isinstance(value, property):
                yield (attr,getattr(self, attr))

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
    # CTC related, if ctc_weight > 0, CTC loss is applied in training.
    ctc_weight: Optional[float] = field(default=0.0)
    # For decode
    decode_instruction: Optional[str] = field(default="")

    @property
    def ds_rate(self):
        return self.encoder_ds_rate * self.encoder_projector_ds_rate

    @ds_rate.setter
    def ds_rate(self, value):
        if value != self.encoder_ds_rate * self.encoder_projector_ds_rate:
            raise ValueError(" ds_rate != encoder_ds_rate / encoder_projector_ds_rate")

    @property
    def speech_tokens_per_second(self):
        return self.frames_per_second / self.ds_rate

    @speech_tokens_per_second.setter
    def speech_tokens_per_second(self, value):
        if value != self.frames_per_second / self.ds_rate:
            raise ValueError(" speech_tokens_per_second != frames_per_second / ds_rate")

    @property
    def max_speech_token_size(self):
        return math.ceil(self.max_speech_seconds *
                         self.speech_tokens_per_second)

    @max_speech_token_size.setter
    def max_speech_token_size(self, value):
        if value != math.ceil(self.max_speech_seconds * self.speech_tokens_per_second):
            raise ValueError(" max_speech_token_size != max_speech_seconds * speech_tokens_per_second")

    @property
    def max_mel_size(self):
        return self.max_speech_seconds * self.frames_per_second

    @max_mel_size.setter
    def max_mel_size(self, value):
        if value != self.max_speech_seconds * self.frames_per_second:
            raise ValueError(" max_mel_size != max_speech_seconds * frames_per_second")

class WestSpeechConfig(PretrainedConfig):
    model_type = "west_speech_model"

    def __init__(self, config:ModelArguments=None, **kwargs):
        super().__init__(**kwargs)
        self._name_or_path = "./west-slm"
        if config is not None:
        # config = ModelArguments()
            for attr, value in config:
                setattr(self, attr, value) 

def ctc_reduce(hyp, blank_id: int = 0):
    new_hyp = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


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

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
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
    config_class = WestSpeechConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        model_args: ModelArguments = None,
        **kwargs
    ):
        super().__init__(config)
        llm_config = transformers.AutoConfig.from_pretrained(
            config.llm_model_name_or_path)
        # llm_config.use_cache = False

        if 1:
        # if model_args is not None:
            if config.encoder_type == "whisper":
                if model_args is not None:
                    encoder = whisper.load_model(config.whisper_model_name_or_path)
                    for field in dataclasses.fields(encoder.dims):
                        setattr(config, f'whisper_{field.name}', getattr(encoder.dims, field.name))
                else:
                    whisper_dim = ModelDimensions(n_mels=1, 
                                                  n_audio_ctx=1 ,
                                                  n_audio_state=1 , 
                                                  n_audio_head=1 , 
                                                  n_audio_layer=1 , 
                                                  n_vocab=1 , 
                                                  n_text_ctx=1 ,
                                                  n_text_state=1 , 
                                                  n_text_head=1 ,
                                                  n_text_layer=1)
                    for field in dataclasses.fields(whisper_dim):
                        v = getattr(config, f'whisper_{field.name}')
                        setattr(whisper_dim, field.name, v)
                    encoder = Whisper(whisper_dim)
                

            elif config.encoder_type == "wenet":
                encoder = wenet.load_model_pt(config.wenet_model_name_or_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                encoder = encoder.to(device)
            else:
                raise ValueError(f"Unexpected encoder type {config.encoder_type}")

            # Load llm model and tokenizer
            llm_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name_or_path,
                config=llm_config,
                torch_dtype='auto',
            )
            if config.encoder_type == "whisper":
                encoder_dim = encoder.dims.n_audio_state
            else:
                encoder_dim = encoder.encoder.output_size()
            
            config.encoder_dim = encoder_dim
        if config.encoder_dim:
            encoder_dim = config.encoder_dim

        config.hidden_size = llm_config.hidden_size
        llm_dim = llm_config.hidden_size
        projector = ProjectorCov1d(config, encoder_dim, llm_dim)
        total_params = sum(p.numel() for p in projector.parameters())
        print('Projector total params: {:.2f}M'.format(total_params / 1024 / 1024))

        self.llm = llm_model
        self.encoder = encoder
        self.projector = projector
        # self._keys_to_ignore_on_save = set()
        # Do not save the parameter of llm and whisper
        # for k in self.llm.state_dict().keys():
        #     self._keys_to_ignore_on_save.add('llm.' + k)
        # for k in self.encoder.state_dict().keys():
        #     self._keys_to_ignore_on_save.add('encoder.' + k)
        # Use bos_token_id as CTC blank id
        self.ctc_loss = nn.CTCLoss(config.bos_token_id,
                                   reduction='mean',
                                   zero_infinity=True)
        self.blank_id = config.bos_token_id
        self.model_args = config

    def get_speech_embeddings(self, mel, mel_len):
        max_speech_size = self.model_args.max_speech_token_size
        if self.model_args.encoder_type == 'whisper':
            speech_emb = self.encoder.embed_audio(mel)  # (b, n_mel, 1500)
            speech_proj = self.projector(speech_emb)
        else:
            mel = mel.transpose(1, 2)
            # mask (B, 1, T)
            speech_emb, mask = self.encoder._forward_encoder(mel, mel_len)
            speech_emb = speech_emb.masked_fill(~mask.transpose(1, 2), 0.0)
            # Note: The downsampling strategy in wenet discards frames that
            # are not enough for an output, so we need to pad the output to
            # a fixed length.
            speech_proj = self.projector(speech_emb)
            pad_size = max_speech_size - speech_proj.size(1)
            speech_proj = F.pad(speech_proj, (0, 0, 0, pad_size), value=0.0)
        return speech_proj

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mel: torch.LongTensor = None,
        mel_len: torch.LongTensor = None,
        ctc_ids: torch.LongTensor = None,
        ctc_ids_len: torch.LongTensor = None,
        logits_to_keep: Union[int, torch.Tensor] = 0
    ):
        max_speech_size = self.model_args.max_speech_token_size
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb = self.get_speech_embeddings(mel, mel_len)
        inputs_embeds = torch.cat(
            (speech_emb, text_emb[:, max_speech_size:, :]), dim=1)
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       labels=labels, 
                       logits_to_keep=logits_to_keep)
        ctc_weight = self.model_args.ctc_weight
        if ctc_weight > 0:
            # Tie CTC linear transforme and input embedding weight
            ctc_linear = self.llm.get_input_embeddings().weight
            ctc_act = torch.matmul(speech_emb, ctc_linear.T)
            ctc_act = ctc_act.transpose(0, 1)
            ctc_prob = ctc_act.log_softmax(2)
            prob_len = torch.ceil(mel_len / self.model_args.ds_rate).long()
            with torch.amp.autocast(enabled=False):
                closs = self.ctc_loss(ctc_prob.float(), ctc_ids, prob_len,
                                      ctc_ids_len)
            out.loss = (1 - ctc_weight) * out.loss + ctc_weight * closs
        return out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        mel: torch.LongTensor = None,
        mel_len: torch.LongTensor = None,
        decode_config=None,
        **kwargs
    ):
        max_speech_size = self.model_args.max_speech_token_size
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb = self.get_speech_embeddings(mel, mel_len)
        inputs_embeds = torch.cat(
            (speech_emb, text_emb[:, max_speech_size:, :]), dim=1)
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=decode_config.do_sample,
            top_p=decode_config.top_p,
            temperature=decode_config.temperature,
            num_beams=decode_config.num_beams,
            max_new_tokens=decode_config.max_new_tokens,
            eos_token_id=decode_config.eos_token_id,
            pad_token_id=decode_config.pad_token_id,
            **kwargs
        )
        return model_outputs

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def decode_ctc(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        mel: torch.LongTensor = None,
        mel_len: torch.LongTensor = None,
        eos_token_id=None,
        decode_config=None,
    ):
        speech_emb = self.get_speech_embeddings(mel, mel_len)
        # Tie CTC linear transforme and input embedding weight
        ctc_linear = self.llm.get_input_embeddings().weight
        ctc_act = torch.matmul(speech_emb, ctc_linear.T)
        ctc_probs = ctc_act.log_softmax(2)
        prob_len = torch.ceil(mel_len / self.model_args.ds_rate).long()
        batch_size = ctc_probs.size(0)
        results = []
        for i in range(batch_size):
            top1 = ctc_probs[i][:prob_len[i], :].argmax(dim=1)
            hyp = ctc_reduce(top1.tolist(), self.blank_id)
            results.append(hyp)
        return results

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_projector(self):
        freeze_model(self.projector)
        # self.projector.eval()

    def freeze_encoder(self):
        freeze_model(self.encoder)
        self.encoder.eval()

    def freeze_llm(self):
        freeze_model(self.llm)

    def load_projector(self, projector_path):
        projector_state_dict = safetensors.torch.load_file(projector_path)
        self.load_state_dict(projector_state_dict, strict=False)


def init_model(model_args):
    west_model_config = WestSpeechConfig(config=model_args)
    model = SpeechLLM(west_model_config, model_args=model_args)
    if model_args.projector_model_path is not None:
        model.load_projector(model_args.projector_model_path,)
    return model

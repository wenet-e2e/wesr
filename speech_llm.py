# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math
from typing import Optional
from dataclasses import dataclass, field

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel
import wenet
import whisper

from utils import ctc_reduce, ctc_peak_time, lists_to_tensor


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
    # CTC related, if ctc_weight > 0, CTC loss is applied in training.
    ctc_weight: Optional[float] = field(default=0.0)
    ctc_reduce: Optional[bool] = field(default=False)
    reduced_speech_token_per_second: int = 6

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
        # Use bos_token_id as CTC blank id
        self.ctc_loss = nn.CTCLoss(config.bos_token_id,
                                   reduction='mean',
                                   zero_infinity=True)
        self.blank_id = config.bos_token_id
        self.model_args = model_args

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

    def select_speech_embeddings(
        self,
        ctc_prob,
        ctc_ids,
        prob_len,
        ctc_ids_len,
        speech_emb,
    ):
        """ Select speech embeddings by force align
        """
        out_emb = torch.zeros_like(speech_emb)
        batch_size = ctc_prob.size(0)
        for i in range(batch_size):
            # The current forced_align only supports batch_size==1.
            alignment, _ = torchaudio.functional.forced_align(
                ctc_prob[i, :prob_len[i], :].contiguous().unsqueeze(0),
                ctc_ids[i, :ctc_ids_len[i]].contiguous().unsqueeze(0),
                blank=self.blank_id)
            peak_t = ctc_peak_time(alignment[0].tolist(), self.blank_id)
            assert ctc_ids_len[i].item() == len(peak_t)
            out_emb[i, :ctc_ids_len[i], :] = speech_emb[i, peak_t, :]
        return out_emb

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
    ):
        max_speech_size = self.model_args.max_speech_token_size
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb = self.get_speech_embeddings(mel, mel_len)
        ctc_weight = self.model_args.ctc_weight
        # Optional, add CTC loss
        if ctc_weight > 0:
            # Tie CTC linear transforme and input embedding weight
            ctc_linear = self.llm.get_input_embeddings().weight
            ctc_act = torch.matmul(speech_emb, ctc_linear.T)
            ctc_act = ctc_act.transpose(0, 1)
            ctc_prob = ctc_act.log_softmax(2)
            prob_len = torch.ceil(mel_len / self.model_args.ds_rate).long()
            with torch.cuda.amp.autocast(enabled=False):
                closs = self.ctc_loss(ctc_prob.float(), ctc_ids, prob_len,
                                      ctc_ids_len)
            # Optional, reduce sequence, rewrite speech embeddings
            if self.model_args.ctc_reduce:
                speech_emb = self.select_speech_embeddings(
                    ctc_prob.transpose(0, 1), ctc_ids, prob_len, ctc_ids_len,
                    speech_emb)
        inputs_embeds = torch.cat(
            (speech_emb, text_emb[:, max_speech_size:, :]), dim=1)
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       labels=labels)
        if ctc_weight > 0:
            out.loss = (1 - ctc_weight) * out.loss + ctc_weight * closs
        return out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mel: torch.LongTensor = None,
        mel_len: torch.LongTensor = None,
        eos_token_id=None,
        decode_config=None,
    ):
        max_speech_size = self.model_args.max_speech_token_size
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb = self.get_speech_embeddings(mel, mel_len)
        # Optional, appliy ctc_reduce
        if self.model_args.ctc_reduce:
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
            results_len = torch.tensor([len(l) for l in results],
                                       dtype=torch.long,
                                       device=mel.device)
            results_ids = lists_to_tensor(results, device=mel.device)
            speech_emb = self.select_speech_embeddings(ctc_probs, results_ids,
                                                       prob_len, results_len,
                                                       speech_emb)
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

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def decode_ctc(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
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

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_similarity(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mel: torch.LongTensor = None,
        mel_len: torch.LongTensor = None,
        ctc_ids: torch.LongTensor = None,
        ctc_ids_len: torch.LongTensor = None,
    ):
        """ Compute text and speech embedding similarity
        """
        max_speech_size = self.model_args.max_speech_token_size
        text_emb = self.llm.get_input_embeddings()(ctc_ids)
        speech_emb = self.get_speech_embeddings(mel, mel_len)
        ctc_linear = self.llm.get_input_embeddings().weight
        ctc_act = torch.matmul(speech_emb, ctc_linear.T)
        ctc_probs = ctc_act.log_softmax(2)
        prob_len = torch.ceil(mel_len / self.model_args.ds_rate).long()
        speech_emb = self.select_speech_embeddings(ctc_probs, ctc_ids, prob_len,
                                                   ctc_ids_len, speech_emb)
        batch_size = ctc_ids.size(0)
        results = []
        for i in range(batch_size):
            end = ctc_ids_len[i]
            s = F.cosine_similarity(text_emb[i, :end, :],
                                    speech_emb[i, :end, :],
                                    dim=1)
            results.append(s)
        return results

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

# Copyright (c) 2024 Binbin Zhang(binbzha@qq.com)
# This code is based on the QWen2 from
# https://github.com/QwenLM/Qwen2/blob/main/examples/sft/finetune.py

import logging
import os
import pathlib
from dataclasses import dataclass, field

import torch
import transformers
from transformers import AutoTokenizer, Trainer
from transformers import TrainerCallback

from dataset import DataArguments, SpeechDataset
from speech_llm import init_model, ModelArguments


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adafactor")


class ProfilerCallback(TrainerCallback):

    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        self.profiler = None
        self.started = False
        self.global_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.global_step = state.global_step

        if not self.started:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=9,
                                                 warmup=0,
                                                 active=1,
                                                 repeat=0,
                                                 skip_first=5),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_modules=True)
            self.profiler.__enter__()
            self.started = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.started and self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    model = init_model(model_args)
    model.freeze_llm()
    model.freeze_encoder()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
    )
    if 'llama' in model_args.llm_model_name_or_path:
        tokenizer.pad_token = '<|finetune_right_pad_id|>'

    print("Loading data...")
    train_dataset = SpeechDataset(data_args.data_path, tokenizer, model_args)
    if data_args.eval_data_path:
        eval_dataset = SpeechDataset(data_args.eval_data_path, tokenizer,
                                     model_args)
    else:
        eval_dataset = None
    # Start trainer
    print(training_args.logging_dir)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda x: x[0],
        callbacks=[ProfilerCallback(log_dir=training_args.logging_dir)],
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()

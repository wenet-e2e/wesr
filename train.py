# Copyright (c) 2024 Binbin Zhang(binbzha@qq.com)
# This code is based on the QWen2 from
# https://github.com/QwenLM/Qwen2/blob/main/examples/sft/finetune.py

import pathlib
from dataclasses import dataclass, field

import torch
import transformers
from transformers import AutoTokenizer, Trainer
from transformers.trainer_utils import seed_worker
from torch.utils.data import Dataset, Sampler, DataLoader

from dataset import DataArguments, SpeechDataset, CustomDataCollator, DynamicBatchSampler
from speech_llm import init_model, ModelArguments


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adafactor")


class CustomTrainer(Trainer):

    def __init__(self,
                 batch_sampler: torch.utils.data.Sampler = None,
                 *args,
                 **kwargs):
        self.batch_sampler = batch_sampler
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        dataloader_params = {
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            if self.batch_sampler is not None:
                dataloader_params["batch_sampler"] = self.batch_sampler
            else:
                dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["batch_size"] = self._train_batch_size
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params[
                "prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_params))


def main():
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
    data_collator = CustomDataCollator(ds_rate=model_args.ds_rate,
                                       pad_token_id=tokenizer.pad_token_id)
    if model_args.max_tokens_in_batch is not None:
        batch_sampler = DynamicBatchSampler(train_dataset,
                                            model_args.max_tokens_in_batch,
                                            model_args.ds_rate)
    else:
        batch_sampler = None
    if data_args.eval_data_path:
        eval_dataset = SpeechDataset(data_args.eval_data_path, tokenizer,
                                     model_args)
    else:
        eval_dataset = None
    # Start trainer
    trainer = CustomTrainer(model=model,
                            tokenizer=tokenizer,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            data_collator=data_collator,
                            batch_sampler=batch_sampler)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()

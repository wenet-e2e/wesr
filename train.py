# Copyright (c) 2024 Binbin Zhang(binbzha@qq.com)
# This code is based on the QWen2 from
# https://github.com/QwenLM/Qwen2/blob/main/examples/sft/finetune.py

import pathlib
from dataclasses import dataclass, field

import transformers
from transformers import AutoTokenizer, Trainer

from dataset import DataArguments, SpeechDataset
from speech_llm import init_model, ModelArguments, SpeechLLM
from trl import GRPOConfig
from speech_grpo_trainer import SpeechGRPOTrainer
from reward_funcs import active_reward_func

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adafactor")
    grpo: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    temperature: float = field(default=0.9)


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
    # model.freeze_encoder()
    

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
    train_dataset = SpeechDataset(data_args.data_path, tokenizer, model_args, grpo = training_args.grpo)
    if data_args.eval_data_path:
        eval_dataset = SpeechDataset(data_args.eval_data_path, tokenizer,
                                     model_args)
    else:
        eval_dataset = None

    if training_args.grpo:
        config = training_args.to_dict()
        config['remove_unused_columns'] = False
        config['num_generations'] = 2  #GRPO中的group number
        del config['grpo']
        grpo_config = GRPOConfig(**config)
        trainer_cls = SpeechGRPOTrainer
        trainer_arg = {
            'processing_class': tokenizer,
            'reward_funcs': active_reward_func
        }
        args = grpo_config
        ref_model = SpeechLLM.from_pretrained('./west-slm')
        ref_model.freeze_llm()
        ref_model.freeze_encoder()
        ref_model.freeze_projector()
        ref_model.eval()
    else:
        trainer_cls = Trainer 
        trainer_arg = {
            'tokenizer': tokenizer,
        }
        # training_args.remove_unused_columns = False
        args = training_args

    print(args.to_dict())
    # Start trainer
    trainer = trainer_cls(model=model, ref_model=ref_model,
                      args=args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      **trainer_arg)
    if 1:
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()

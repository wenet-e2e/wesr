from accelerate import Accelerator
from speech_llm import init_model, ModelArguments
import transformers
from dataset import DataArguments
from train import TrainingArguments
from trl.models import unwrap_model_for_generation

def export_model(model, output_dir):
    accelerator = Accelerator()
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        unwrapped_model.save_pretrained(output_dir)

if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments,TrainingArguments ))
    (
        model_args,
        data_args,
        _
    ) = parser.parse_args_into_dataclasses()

    model = init_model(model_args)
    model.freeze_llm()
    export_model(model, './west-slm')
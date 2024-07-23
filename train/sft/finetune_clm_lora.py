#!/usr/bin/env python
# coding=utf-8
# This code is based on Llama-Chinese: github.com/LlamaFamily/Llama-Chinese

import logging
import math
import os
import sys
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain
import deepspeed
from typing import Optional,List,Union,Dict
import numpy as np
import datasets
import evaluate
import torch
from datasets import load_dataset, load_metric
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from torch.utils.data.sampler import WeightedRandomSampler
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import pdb


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    target_modules: Optional[str] = field(
        default='q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj',
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    load_in_bits: Optional[int] = field(default=8)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
        if type(self.target_modules)==str:
            self.target_modules = self.target_modules.split(',')


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_type: str = field(default='number of injuried people')
    train_on_inputs: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_files: Optional[List[str]]  = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_files is None and self.validation_files is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            # kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

            # only save trainable model parameters
            params_checkpoint = OrderedDict()
            for k, v in kwargs["model"].named_parameters():
                if v.requires_grad:
                    params_checkpoint[k] = v.detach().cpu()
            torch.save(params_checkpoint, f'{checkpoint_folder}/pytorch_model.pt')
            return control
        
def load_model(model, path):
    save_model = torch.load(os.path.join(path, 'pytorch_model.pt'))
    model_dict =  model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded model from {path}")    
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"Loaded tokenizer from {path}")
    return tokenizer

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        if model.get_output_embeddings() is not None:
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

meta_data = {
    "task_type": ["number_of_injuried_people", "severity", "accident_type"],
    "description": {
        "number_of_injuried_people": "number of injuried people",
        "severity": "severity",
        "accident_type": "accident type"
    },
    "special_tokens":{
        "number_of_injuried_people": ["<ZERO>", "<ONE>", "<TWO>", "<THREE OR MORE THAN THREE>"],
        "severity": ["<NO APPARENT INJURY>", "<POSSIBLE INJURY>", "<MINOR INJURY>", "<SERIOUS INJURY>", "<FATAL>"],
        "accident_type": ["<SINGLE VEHICLE WITH OBJECT>", "<ANGLE IMPACTS_RIGHT>", "<OTHER>", "<SIDESWIPES_LEFT>", "<FRONT END COLLISIONS>", "<REAR END COLLISIONS>", "<OVERTURN>", "<ANIMAL COLLISIONS>", "<PEDESTRIAN COLLISIONS>", "<SIDESWIPES_RIGHT>", "<PEDALCYCLIST COLLISIONS>", "<HEAD ON COLLISIONS>", "<OFF ROAD>", "<ANGLE IMPACTS_LEFT>" ],

    },
    "tokens_explanation":{
        "number_of_injuried_people": "[<ZERO>: zero, <ONE>: one, <TWO>: two, <THREE OR MORE THAN THREE>: three or more than three]",
        "severity": "[<NO APPARENT INJURY>: no apparent injury, <POSSIBLE INJURY>: possible injury, <MINOR INJURY>: minor injury,  <SERIOUS INJURY>: serious injury, <FATAL>: fatal]",
        "accident_type": "[<SINGLE VEHICLE WITH OBJECT>: single vehicle with object, <ANGLE IMPACTS_RIGHT>: angle impacts_right, <OTHER>: others, <SIDESWIPES_LEFT>: sideswipes_left, <FRONT END COLLISIONS>: front end collisions, <REAR END COLLISIONS>: rear end collisions, <OVERTURN>: overturn, <ANIMAL COLLISIONS>: animal collisions, <PEDESTRIAN COLLISIONS>: pedestrian collisions, <SIDESWIPES_RIGHT>: sideswipes right, <PEDALCYCLIST COLLISIONS>: pedalcyclist collisions, <HEAD ON COLLISIONS>: head on collisions, <OFF ROAD>: off road, <ANGLE IMPACTS_LEFT>: angle impacts left]"
    }
}

class ACCTrainer(Trainer):
    # rebalanced sampler for single gpu
    def _get_train_sampler(self):
        generator = None
        # torch version should > 1.6
        if self.args.world_size <= 1:
            generator = torch.Generator()
            generator.manual_seed(self.args.seed)
            
        counts_of_classes = {}
        for data_item in self.train_dataset:
            cur_class = data_item['labels'][-1]
            if cur_class not in counts_of_classes.keys():
                counts_of_classes[cur_class] = 1
            else:
                counts_of_classes[cur_class] += 1
        classes_weights = dict([(k, len(self.train_dataset)/(v+1)) for k,v in counts_of_classes.items()])
        print(f'counts_of_classes: {counts_of_classes}')
        print(f'classes_weights: {classes_weights}')
        weights = [classes_weights[data_item['labels'][-1]] for data_item in self.train_dataset]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        return sampler
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # pdb.set_trace()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    only_eval = training_args.do_eval and not training_args.do_train

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if True:
        data_files = {}
        dataset_args = {}
        if data_args.train_files is not None:
            data_files["train"] = data_args.train_files
        if data_args.validation_files is not None:
            data_files["validation"] = data_args.validation_files
        extension = (
            data_args.train_files[0].split(".")[-1]
            if data_args.train_files is not None
            else data_args.validation_files.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=os.path.join(training_args.output_dir,'dataset_cache'),
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )

        task_type = data_args.task_type
        assert task_type in meta_data['task_type']
        description = meta_data['description'][task_type]
        special_tokens = meta_data['special_tokens'][task_type]
        tokens_explanation = meta_data['tokens_explanation'][task_type]

        system_prompt = f'System: You are a helpful assistant to predict the traffic accident.' + \
            f'Please predict the {description} of the accident choosing from the following tokens: {tokens_explanation}. \n User: '
        gpt_prompt = 'Assistant: The answer is: '

        raw_datasets["train"] = raw_datasets["train"].map(lambda x:{
            'input': system_prompt + x['text'].split('Assistant:')[0].split('Human:')[1] + gpt_prompt,
            'target': x['text'].split('Assistant:')[1][:-4].strip()
        }, remove_columns=['text'])
        raw_datasets["validation"] = raw_datasets["validation"].map(lambda x:{
            'input': system_prompt + x['text'].split('Assistant:')[0].split('Human:')[1] +  gpt_prompt,
            'target': x['text'].split('Assistant:')[1][:-4].strip()
        }, remove_columns=['text'])
        print('raw_datasets["train"]', raw_datasets["train"][:2])
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "padding_side":'left'
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules =  model_args.target_modules,
        fan_in_fan_out = False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(lora_config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        print(torch_dtype)
        torch_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            load_in_8bit=True if model_args.load_in_bits==8 else False,
            trust_remote_code=True,
            use_flash_attention_2=True,
            quantization_config=bnb_config if model_args.load_in_bits==4 else None,
            # device_map  = 'auto'
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
        )
        # model = prepare_model_for_int8_training(model, output_embedding_layer_name="embed_out", layer_norm_names=[])

    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # add class tokens

    smart_tokenizer_and_embedding_resize(
            special_tokens_dict={'additional_special_tokens': special_tokens},
            tokenizer=tokenizer,
            model=model,
    )


    if model_args.load_in_bits==8:
        model = prepare_model_for_int8_training(model)
    elif model_args.load_in_bits==4:
        model = prepare_model_for_kbit_training(model)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    train_on_inputs = True
    if len(column_names)==1:
        text_column_name = "text" if "text" in column_names else column_names[0]
    elif len(column_names)==2:
        input_column_name = 'input' if 'input' in column_names else column_names[0]
        target_column_name = 'target' if 'target' in column_names else column_names[0]
        train_on_inputs=False
    else:
        raise ValueError('输入文件列数不对')
    print('train_on_inputs',train_on_inputs)
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer([ item for item in examples[text_column_name]],truncation=True,max_length=data_args.block_size,padding=False,return_tensors=None)
            output['labels'] = output['input_ids'].copy()
        return output

    def tokenize(prompt):
        result = tokenizer(prompt,truncation=True,max_length=data_args.block_size,padding=False,return_tensors=None)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        input_text = data_point[input_column_name]
        target_text = data_point[target_column_name]
        full_prompt = input_text+target_text
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = input_text
            tokenized_user_prompt = tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt



    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function if train_on_inputs==True else generate_and_tokenize_prompt,
                batched=True if train_on_inputs==True else False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function if train_on_inputs==True else generate_and_tokenize_prompt,
                batched=True if train_on_inputs==True else False,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            block_size = 2048
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric_acc = evaluate.load("accuracy.py")
        metric_f1 = load_metric("f1")
        metric_pre = load_metric("precision")
        metric_rec = load_metric("recall")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:]
            preds = preds[:, :-1]
            true_predictions = [
                [p for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(preds, labels)
            ]
            true_labels = [
                [l for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(preds, labels)
            ]
            true_predictions = [x[-1] for x in true_predictions]
            true_labels = [x[-1] for x in true_labels]
            preds = np.array(true_predictions).reshape(-1)
            labels = np.array(true_labels).reshape(-1)
            return {
                'accuracy': metric_acc.compute(predictions=preds, references=labels)['accuracy'],
                'f1': metric_f1.compute(predictions=preds, references=labels, average="macro")['f1'],
                'precision': metric_pre.compute(predictions=preds, references=labels, average="macro")['precision'],
                'recall': metric_rec.compute(predictions=preds, references=labels, average="macro")['recall'],
            }

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if only_eval:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        tokenizer = load_model(model, resume_from_checkpoint)
    else:
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = True

    # Initialize our Trainer
    trainer = ACCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None)
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            resume_from_checkpoint = training_args.resume_from_checkpoint
            if os.path.exists(resume_from_checkpoint):
                print(f"Restarting from {resume_from_checkpoint}")
                tokenizer = load_model(model, resume_from_checkpoint)
            else:
                print(f"Checkpoint {resume_from_checkpoint} not found")
            # checkpoint = Fa
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

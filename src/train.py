from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from train.arguments import (
    ModelArguments,
    DataTrainingArguments,
    PeftArguments,
)

from peft import get_peft_model, LoraConfig, TaskType

from common.setup_logger import setup_logger

from dataclasses import dataclass, field
from transformers import HfArgumentParser

import os
import torch
from peft import PeftModel

from datasets import load_dataset
from train.data_processors.data_preprocessors import Preprocessor

import numpy as np

import logging

def load_lora_checkpoint(model,checkpoint_path, logger=None, merge=False):
    adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, checkpoint_path)
        if logger:
            logger.info(f"load checkpoint (adapter) from: {checkpoint_path}")
    else:
        sd_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(sd_path):
            model.load_state_dict(
                torch.load(sd_path), 
                strict=False
            )
            if logger:
                logger.info(f"load checkpoint (state_dict) from: {checkpoint_path}")
        else:
            if logger:
                logger.error(f"no checkpoint found at: {checkpoint_path}")
    
    if merge:
        model = model.merge_and_unload()
        
    return model

def remove_minus100(ids,val):
    """
        -100是HF预留的id（不参与loss计算）
        有的tokenizer在decode -100时会报错
        因此在decode之前去除（替换为pad_id）
    """
    ids = np.array(ids)
    ids = np.where(ids == -100, val, ids)
    return ids

def print_dataset_example(example,tokenizer):
    print("input_ids",example["input_ids"])
    print("inputs", tokenizer.decode(example["input_ids"],skip_special_tokens=True))
    print("label_ids", example["labels"])
    label_ids = remove_minus100(example["labels"],tokenizer.pad_token_id)
    print("labels", tokenizer.decode(label_ids,skip_special_tokens=True))

def load_raw_datasets(data_args,cache_dir):
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    # 加载数据集
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=cache_dir
    )

    return raw_datasets

def load_model(model_name):
    # 加载ChatGLM的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 加载模型
    model = None
    if False:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        model = model.half()
        model.is_parallelizable = True
        model.model_parallel = True
    return model, tokenizer

def create_peft_config(peft_args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=peft_args.lora_rank,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=["query_key_value"],
    )
    return peft_config

def main():

    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, Seq2SeqTrainingArguments))

    model_args, data_training_args, peft_args, seq2seq_training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_training_args)
    print(peft_args)
    print(seq2seq_training_args)
    
    print(seq2seq_training_args.get_process_log_level())
    setup_logger(seq2seq_training_args.get_process_log_level())

    logger = logging.getLogger(__name__)

    set_seed(seq2seq_training_args.seed)

    model, tokenizer = load_model(model_args.model_name_or_path)

    if peft_args.lora_checkpoint is not None:
        model = load_lora_checkpoint(
            model,
            peft_args.lora_checkpoint,
            logger=logger,
            merge=not seq2seq_training_args.do_train
        )
        model = model.cuda()


    # 加载数据集
    raw_datasets = load_raw_datasets(data_training_args,model_args.cache_dir)

    data_processor = Preprocessor(
        data_args=data_training_args,
        tokenizer=tokenizer
    )

    if seq2seq_training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif seq2seq_training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif seq2seq_training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    print(column_names)

    if seq2seq_training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        # 随机排序训练集
        train_dataset = raw_datasets["train"].shuffle(seq2seq_training_args.seed)
        
        with seq2seq_training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                data_processor.preprocess_function_train,
                batched=True,
                num_proc=data_training_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_training_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0],tokenizer)


    # Data collator
    label_pad_token_id = -100 if data_training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None, 
        padding=False
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the unmodified base models with the CRF training metrics."""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch.distributed as dist
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from src.training.config import ModelArguments
from src import prompts_list



@dataclass
class ZeroShotArguments:
    val_data_path: str = "anonymousubmission/notes_with_related_items"
    val_data_split: str = "dev"
    test_data_split: str = "test"
    max_n_examples_val: Optional[int] = None
    max_n_examples_test: Optional[int] = None
    cache_dir: Optional[str] = None
    task_type_list: List[str] = field(
        default_factory=lambda: ["setting_0", "setting_1", "setting_2"]
    )
    setting_for_example_in_setting_3: str = ''


def parse_args():
    parser = HfArgumentParser(
        (ModelArguments, Seq2SeqTrainingArguments, ZeroShotArguments),
        description=__doc__,
    )
    # Defaults copied from train_configs/crf_task/llama1B.yaml. The YAML is
    # deliberately not parsed by this evaluator.
    parser.set_defaults(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        hub_model_id="your_username/crf-gamma-Llama-3.2-1B-Instruct",
        torch_dtype="bfloat16",
        use_lora=True,
        quantization=None,
        force_auto_device_map=False,
        use_flash_attention=False,
        deepspeed=None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        do_train=False,
        do_eval=True,
        do_predict=True,
        per_device_eval_batch_size=4,
        logging_first_step=True,
    )
    # HfArgumentParser marks this dataclass field as required at construction
    # time; set_defaults alone does not clear argparse's required flag.
    for action in parser._actions:
        if action.dest == "output_dir":
            action.required = False
    return parser.parse_args_into_dataclasses()


def assistant_start(model_name):
    name = model_name.lower()
    if "qwen" in name:
        return "<|im_end|>\n<|im_start|>assistant\n"
    if "gemma-3" in name or "medgemma" in name:
        return "<start_of_turn>model\n"
    if "gemma-4" in name:
        return "<|turn>model\n"
    if "llama" in name:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if "unsup" in name and "phi" in name:
        return "<|end|>\n<|assistant|>\n"
    if "phi" in name:
        return "<|end|><|assistant|>\n"
    raise ValueError(f"Cannot determine the assistant marker for {model_name}.")



def main():
    model_args, eval_args, args = parse_args()
    # These repository modules initialize HF-backed resources at import time;
    # keep them after parsing so `--help` remains an offline operation.
    from src.training.dataset import DataCollatorForMedLlm, MedLlmDataset
    from src.training.evaluation_functions import (
        compute_metrics_crf_gamma,
        preprocess_logits_for_metrics,
    )
    from src.training.load_model import load_model
    from src.training.trainer import MedLlmTrainer

    logging.basicConfig(level=logging.INFO)

    # Preserve the YAML-derived CLI defaults above, then apply only the changes
    # required by zero-shot evaluation.
    model_args.use_lora = False
    model_args.quantization = None
    eval_args.do_train = False
    eval_args.do_eval = True
    # Training YAMLs schedule evaluation every N steps. There is no training
    # loop in this script; validation and test are evaluated explicitly below.
    eval_args.eval_strategy = "no"
    eval_args.gradient_checkpointing = False
    eval_args.label_names = ["labels", "loss_weight_mask"]
    eval_args.deepspeed = None

    model, tokenizer = load_model(
        inference=True,
        model_weights_name_or_path=model_args.model_name_or_path,
        quantization=None,
        use_lora=False,
        torch_dtype=model_args.torch_dtype,
        force_auto_device_map=model_args.force_auto_device_map,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=args.cache_dir,
        use_flash_attention=model_args.use_flash_attention,
        max_memory_MB=model_args.max_memory_MB,
        rope_scaling_factor=model_args.rope_scaling_factor,
        max_seq_length=model_args.max_seq_length,
    )

    marker_ids = tokenizer(
        assistant_start(model_args.model_name_or_path), add_special_tokens=False
    )["input_ids"]
    special_ids = list(tokenizer.added_tokens_decoder.keys())
    if "Qwen3" in model_args.model_name_or_path:
        special_ids.append(198)

    for i, SYSTEM_PROMPT in enumerate(prompts_list):
        for task_type in args.task_type_list:

            if len(args.setting_for_example_in_setting_3) > 0 and task_type != 'setting_3':
                raise ValueError("setting_for_example_in_setting_3 should only be set when task_type is 'setting_3'.")
            
            
            logging.info(f"\n\n\n\n-------Evaluating with system prompt {i} and task type {task_type}---------\n\n\n")
            if task_type == 'setting_3':
                task_type_complete = f"{task_type}_0_shot_{model_args.model_name_or_path.split('/')[-1]}_{args.setting_for_example_in_setting_3}_system_prompt_{i}_prediction"
                print(task_type_complete)
            else:
                task_type_complete = task_type + '_0_shot'
            output_dir = os.path.join(
                    "data", "zero_shot", model_args.model_name_or_path.split("/")[-1], task_type, f"system_prompt_{i}"
                )
            eval_args.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            dataset_kwargs = dict(
                tokenizer=tokenizer,
                train_data_path=args.val_data_path,
                training_type=task_type_complete,
                system_prompt=SYSTEM_PROMPT,
                calculate_loss_on_prompt=True,
            )
            validation = MedLlmDataset(
                train_data_split=args.val_data_split,
                max_n_examples=args.max_n_examples_val,
                **dataset_kwargs,
            )
            test = MedLlmDataset(
                train_data_split=args.test_data_split,
                max_n_examples=args.max_n_examples_test,
                **dataset_kwargs,
            )
            if validation.crf_items is None or test.crf_items is None:
                raise ValueError("Evaluation splits must contain a `crf_item` column.")

            calc_per_item = model_args.model_name_or_path not in (
                "microsoft/Phi-3.5-mini-instruct",
                "microsoft/MediPhi",
            )
            current_items = validation.crf_items
            current_prediction_records = []

            def metrics(prediction):
                return compute_metrics_crf_gamma(
                    prediction,
                    marker_ids,
                    special_ids,
                    current_items,
                    [],
                    calc_per_item=calc_per_item,
                    tokenizer=tokenizer,
                    prediction_records=current_prediction_records,
                )

            trainer = MedLlmTrainer(
                model=model,
                processing_class=tokenizer,
                args=eval_args,
                compute_metrics=metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                data_collator=DataCollatorForMedLlm(tokenizer=tokenizer),
                eval_on_start=True
            )

            validation_metrics = trainer.evaluate(validation, metric_key_prefix="validation")
            validation_predictions = list(current_prediction_records)
            current_items = test.crf_items
            current_prediction_records.clear()
            test_metrics = trainer.evaluate(test, metric_key_prefix="test")
            test_predictions = list(current_prediction_records)

            if trainer.is_world_process_zero():
                for filename, values in (
                    ("validation_metrics.json", validation_metrics),
                    ("test_metrics.json", test_metrics),
                ):
                    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
                        json.dump(values, handle, indent=2)
                for filename, records in (
                    ("validation_predictions.json", validation_predictions),
                    ("test_predictions.json", test_predictions),
                ):
                    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
                        json.dump(records, handle, indent=2, ensure_ascii=False)
                logging.info("Zero-shot metrics and predictions saved to %s", output_dir)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()


if __name__ == "__main__":
    main()

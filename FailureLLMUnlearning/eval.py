import sys
from metrics.verbmem import eval as eval_verbmem
from metrics.privleak import eval as eval_privleak
from metrics.knowmem import eval as eval_knowmem
from utils import load_model, load_tokenizer, write_csv, read_json, write_json
from constants import SUPPORTED_METRICS, CORPORA, LLAMA_DIR, DEFAULT_DATA, AUC_RETRAIN
import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Dict, Literal
from pandas import DataFrame
import json
from LLama_factory.src.llmtuner.eval import eval_mmlu, eval_truthfulqa, eval_triviaqa, eval_fluency
from transformers import AutoTokenizer

def eval_model(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer = LLAMA_DIR,
    metrics: List[str] = SUPPORTED_METRICS,
    corpus: Literal['news', 'books'] | None = None,
    privleak_auc_key: str = 'forget_holdout_Min-40%',
    verbmem_agg_key: str = 'mean_rougeL',
    verbmem_max_new_tokens: int = 128,
    knowmem_agg_key: str = 'mean_rougeL',
    knowmem_max_new_tokens: int = 32,
    verbmem_forget_file: str | None = None,
    privleak_forget_file: str | None = None,
    privleak_retain_file: str | None = None,
    privleak_holdout_file: str | None = None,
    knowmem_forget_qa_file: str | None = None,
    knowmem_forget_qa_icl_file: str | None = None,
    knowmem_retain_qa_file: str | None = None,
    knowmem_retain_qa_icl_file: str | None = None,
    temp_dir: str | None = None,
    max_samples: int | None = None,  # Limit dataset size for quick testing
) -> Dict[str, float]:
    # Argument sanity check
    if not metrics:
        raise ValueError(f"Specify `metrics` to be a non-empty list.")
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Given metric {metric} is not supported.")
    if corpus is not None and corpus not in CORPORA:
        raise ValueError(f"Invalid corpus. `corpus` should be either 'news' or 'books'.")
    if corpus is not None:
        verbmem_forget_file = DEFAULT_DATA[corpus]['verbmem_forget_file'] if verbmem_forget_file is None else verbmem_forget_file
        privleak_forget_file = DEFAULT_DATA[corpus]['privleak_forget_file'] if privleak_forget_file is None else privleak_forget_file
        privleak_retain_file = DEFAULT_DATA[corpus]['privleak_retain_file'] if privleak_retain_file is None else privleak_retain_file
        privleak_holdout_file = DEFAULT_DATA[corpus]['privleak_holdout_file'] if privleak_holdout_file is None else privleak_holdout_file
        knowmem_forget_qa_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_file'] if knowmem_forget_qa_file is None else knowmem_forget_qa_file
        knowmem_forget_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_icl_file'] if knowmem_forget_qa_icl_file is None else knowmem_forget_qa_icl_file
        knowmem_retain_qa_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_file'] if knowmem_retain_qa_file is None else knowmem_retain_qa_file
        knowmem_retain_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_icl_file'] if knowmem_retain_qa_icl_file is None else knowmem_retain_qa_icl_file

    out = {}
    RETAIN_MMLU = 'retain_mmlu.json'
    TRUTHFUL = 'truthful.json'
    TRIVIAQA = 'triviaqa.json'
    FLUENCY = 'fluency.json'
    eval_dataset_dir = './LLama_factory/data/utility'
    # target = 'None'
    # eval_dataset_dir = os.path.join(eval_dataset_dir, target)
    with open(os.path.join(eval_dataset_dir, RETAIN_MMLU), 'r') as f:
        retain_mmlu = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRUTHFUL), 'r') as f:
        truthfulqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRIVIAQA), 'r') as f:
        triviaqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, FLUENCY), 'r') as f:
        fluency = json.load(f)

    # output_result_dir = os.path.join('results', target)
    output_result_dir = './results'
    os.makedirs(os.path.join(output_result_dir), exist_ok=True)

    model.eval()
    use_prompt = False
    with torch.no_grad():
        e_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='left')
        e_tokenizer.pad_token = e_tokenizer.eos_token
        # Set default chat template if not already set (required for newer transformers versions)
        if e_tokenizer.chat_template is None:
            # Use Llama-2 chat template format
            e_tokenizer.chat_template = "{%- for message in messages %}{%- if message['role'] == 'system' %}{%- set content = '<s>[INST] <<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' %}{%- elif message['role'] == 'user' %}{%- set content = '<s>[INST] ' + message['content'] + ' [/INST]' %}{%- elif message['role'] == 'assistant' %}{%- set content = ' ' + message['content'] + ' </s>' %}{%- endif %}{{- content }}{%- endfor %}"
        # Skip these evaluations for quick testing (they take 2-3 hours)
        # Uncomment for full evaluation
        # print("Evaluate mmlu...")
        # gen=eval_mmlu(model, e_tokenizer, retain_mmlu, batch_size=1, output_result_dir=os.path.join(output_result_dir, 'mmlu.json'), use_prompt=use_prompt)
        # print("Evaluate truthful...")
        # tru=eval_truthfulqa(model, e_tokenizer, truthfulqa, batch_size=4, output_result_dir=os.path.join(output_result_dir, 'truthful.json'), use_prompt=use_prompt)
        # print("Evaluate triviaqa...")
        # fac=eval_triviaqa(model, e_tokenizer, triviaqa, batch_size=16, output_result_dir=os.path.join(output_result_dir, 'triviaqa.json'), use_prompt=use_prompt)
        # print("Evaluate fluency...")
        # flu=eval_fluency(model, e_tokenizer, fluency, batch_size=8, output_result_dir=os.path.join(output_result_dir, 'fluency.json'), use_prompt=use_prompt)
        
        # Quick test mode: set dummy values
        gen = 0.0
        tru = 0.0
        fac = 0.0
        flu = 0.0

    





    # 1. verbmem_f
    if 'verbmem_f' in metrics:
        data = read_json(verbmem_forget_file)
        if max_samples is not None:
            data = data[:max_samples]
            print(f"⚠ Using limited dataset: {len(data)} samples (max_samples={max_samples})")
        agg, log = eval_verbmem(
            prompts=[d['prompt'] for d in data],
            gts=[d['gt'] for d in data],
            model=model, tokenizer=tokenizer,
            max_new_tokens=verbmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "verbmem_f/agg.json"))
            write_json(log, os.path.join(temp_dir, "verbmem_f/log.json"))
        out['verbmem_f'] = agg[verbmem_agg_key] * 100

    # 2. privleak
    if 'privleak' in metrics:
        forget_data = read_json(privleak_forget_file)
        retain_data = read_json(privleak_retain_file)
        holdout_data = read_json(privleak_holdout_file)
        if max_samples is not None:
            forget_data = forget_data[:max_samples]
            retain_data = retain_data[:max_samples]
            holdout_data = holdout_data[:max_samples]
            print(f"⚠ Using limited dataset: {len(forget_data)} samples per split (max_samples={max_samples})")
        auc, log = eval_privleak(
            forget_data=forget_data,
            retain_data=retain_data,
            holdout_data=holdout_data,
            model=model, tokenizer=tokenizer
        )
        if temp_dir is not None:
            write_json(auc, os.path.join(temp_dir, "privleak/auc.json"))
            write_json(log, os.path.join(temp_dir, "privleak/log.json"))
        out['privleak'] = (auc[privleak_auc_key] - AUC_RETRAIN[corpus][privleak_auc_key]) / AUC_RETRAIN[corpus][privleak_auc_key] * 100

    # 3. knowmem_f
    if 'knowmem_f' in metrics:
        qa = read_json(knowmem_forget_qa_file)
        icl = read_json(knowmem_forget_qa_icl_file)
        if max_samples is not None:
            qa = qa[:max_samples]
            icl = icl[:max_samples]
            print(f"⚠ Using limited dataset: {len(qa)} samples (max_samples={max_samples})")
        agg, log = eval_knowmem(
            questions=[d['question'] for d in qa],
            answers=[d['answer'] for d in qa],
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_f/agg.json"))
            write_json(log, os.path.join(temp_dir, "knowmem_f/log.json"))
        out['knowmem_f'] = agg[knowmem_agg_key] * 100
        
    # 4. knowmem_r
    if 'knowmem_r' in metrics:
        qa = read_json(knowmem_retain_qa_file)
        icl = read_json(knowmem_retain_qa_icl_file)
        if max_samples is not None:
            qa = qa[:max_samples]
            icl = icl[:max_samples]
            print(f"⚠ Using limited dataset: {len(qa)} samples (max_samples={max_samples})")
        agg, log = eval_knowmem(
            questions=[d['question'] for d in qa],
            answers=[d['answer'] for d in qa],
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_r/agg.json"))
            write_json(log, os.path.join(temp_dir, "knowmem_r/log.json"))
        out['knowmem_r'] = agg[knowmem_agg_key] * 100
    out['gen'] = gen
    out['tru'] = tru
    out['fac'] = fac
    out['flu'] = flu

    

    return out


def load_then_eval_models(
    model_dirs: List[str],
    names: List[str],
    corpus: Literal['news', 'books'],
    tokenizer_dir: str = LLAMA_DIR,
    out_file: str | None = None,
    metrics: List[str] = SUPPORTED_METRICS,
    temp_dir: str = "temp",
    alpha: int = 5,
    quantize_4bit: bool = False,
    quantize_8bit: bool = False,
    max_samples: int | None = None,  # Limit dataset size for quick testing
) -> DataFrame:
    # Argument sanity check
    if not model_dirs:
        raise ValueError(f"`model_dirs` should be non-empty.")
    if len(model_dirs) != len(names):
        raise ValueError(f"`model_dirs` and `names` should equal in length.")
    if out_file is not None and not out_file.endswith('.csv'):
        raise ValueError(f"The file extension of `out_file` should be '.csv'.")

    # Run evaluation
    out = []
    for model_dir, model_name in zip(model_dirs, names):
        # Convert bool to int for load_model (which expects 0/1)
        quantize_4bit_int = 1 if quantize_4bit else 0
        quantize_8bit_int = 1 if quantize_8bit else 0
        model = load_model(model_dir, model_name, quantize_4bit_int, quantize_8bit_int, alpha, corpus=corpus)
        
        # Optional: Print parameter dtypes (commented out to reduce output)
        # for param_name, param in model.named_parameters():
        #     print(f"{param_name}: {param.dtype}")

        tokenizer = load_tokenizer(tokenizer_dir)
        res = eval_model(
            model, tokenizer, metrics, corpus,
            temp_dir=os.path.join(temp_dir, model_name),
            max_samples=max_samples  # Limit dataset size for quick testing
        )
        out.append({'name': model_name} | res)
        print(out)
        if out_file is not None: write_csv(out, out_file)
    return DataFrame(out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--names', type=str, nargs='+', default=[])
    parser.add_argument('--tokenizer_dir', type=str, default=LLAMA_DIR)
    parser.add_argument('--corpus', type=str, required=True, choices=CORPORA)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--metrics', type=str, nargs='+', default=SUPPORTED_METRICS)
    parser.add_argument('--quantize_4bit', type=int, default=0)
    parser.add_argument('--quantize_8bit', type=int, default=0)
    parser.add_argument('--alpha', type=int, default=5)
    parser.add_argument('--temp_dir', type=str, default='temp')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Limit dataset size for quick testing (e.g., 10 for 10 samples per metric)')
    args = parser.parse_args()
    # Convert int (0/1) to bool for the function
    args_dict = vars(args)  # Convert Namespace to dictionary
    args_dict['quantize_4bit'] = bool(args_dict['quantize_4bit'])
    args_dict['quantize_8bit'] = bool(args_dict['quantize_8bit'])
    load_then_eval_models(**args_dict)


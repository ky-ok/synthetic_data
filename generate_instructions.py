import time
import json
import random
import re 
import string
import os

import numpy as np 
import tqdm
from rouge import Rouge 

import fire 

import torch
from torch import cuda, bfloat16
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

import logging

import utils as utils

logging.basicConfig(level=logging.INFO)

# tokenizerのdead lock warning を回避
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 文字列から単語を抜き出す
def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

# 文字の置き換え
def replace_keywords(text):
    replacements = {
        "指令：": "1. Instruction:",
        "輸入：": "1. Input:",
        "輸出：": "1. Output:"
    }
    
    for key, value in replacements.items():
        text = text.replace(key, value)
        
    return text

# prompt作成
def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt_en_for_jp.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"######\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"######\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

# Responseチェック
def post_process_response(response):
    """
    Args:
        response: decode済みのresponse
    """
    if response is None:
        return []
    
    print(f"response: {response}")
    
    instructions = []
    
    pattern = fr"\d+\.\s*(Instruction|Input|Output):"
    splitted_data = re.split(pattern, response)
    
    print(f"splitted_data: {splitted_data}")
    print(f"splitted_data_len: {len(splitted_data)}")
    
    if len(splitted_data) > 3 and splitted_data[3] == "Instruction":
        splitted_data = splitted_data[2:] # [0],[1]を削除し、indexを前にずらす
        
        print(f"Updated splitted_data: {splitted_data}")
    
    try:
      inst = splitted_data[2].strip()
      input = splitted_data[4].strip()
      input = "" if input.lower() == "<noinput>" else input
      output = splitted_data[6].strip()
    except IndexError as e:
      print(f"Error: {e}")
      inst = "<noinst>"
      input = "<noinput>"
      output = "<nooutput>"

    # filter based on keywords that are not suitable for language models.
    blacklist = [
        "image",
        "images",
        "graph",
        "graphs",
        "picture",
        "pictures",
        "file",
        "files",
        "map",
        "maps",
        "draw",
        "plot",
        "go to",
        "video",
        "audio",
        "music",
        "flowchart",
        "diagram",            
    ]
    blacklist += []
    if any(find_word_in_string(word, inst) for word in blacklist):
        print("Filter: based on keywords that are not suitable for language models.")
    
    if inst.startswith("Write a program"):
        print("Filter: Write a program is というテキストからInstructionが始まっています")
            
    # filter those starting with punctuation
    # 日本語の場合、""のような特殊文字が最初に来る場合もあるはず
    if inst[0] in string.punctuation.replace("'", "").replace('"', ""):
        print("Filter: 句読点で始まっています")
            
    #  instが全て英語のASCII文字である場合
    if all(char.isascii() for char in inst):
        print("Filter: instは全て英語のASCII文字です")
    # instに英語と日本語意外の文字が含まれている場合
    if any(not (char.isascii() or utils.is_japanese(char)) for char in inst):
        print("Filter: instに英語と日本語意外の文字が含まれています") 
        
    new_inst = {"instruction": inst, "input": input, "output": output}
    print(f"new_inst: {new_inst}")

    instructions.append(new_inst) 
    
    return instructions

    # Generate Instruction
def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks/Elyza-tasks-100_seed_tasks.jsonl",
    num_instructions_to_generate=5,
    model_id="Qwen/Qwen2.5-7B-Instruct",
    num_prompt_instructions=3,
    generate_file_name="regen_elyza.json",
    token="",
    request_batch_size=1,
    temperature=0.8,
    top_p=0.8,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    
    # Load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, generate_file_name)):
        machine_instruction_data = utils.jload(os.path.join(output_dir, generate_file_name))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))
        
    #  first we tokenize all ther seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    
    # Modelの設定
    # 量子化設定
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # モデルの設定
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
        quantization_config=bnb_config, # 量子化
        device_map='auto',
        torch_dtype=torch.bfloat16  
    )
    
    # tokenizerの設定
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="right",
        add_eos_token=True    
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Rougeオブジェクト作成
    rouge = Rouge()
    
    request_idx = 0
    while request_idx < num_instructions_to_generate: 
        request_idx += 1
        
        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
            
        inputs = tokenizer(
            batch_inputs,
            return_tensors="pt"
        ).to(model.device)
        
        terminators = [tokenizer.eos_token_id] # Remove tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
        attention_mask = torch.ones(tuple(inputs['input_ids'].shape), dtype=torch.long).cuda()
        
        request_start = time.time()
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
        
        request_duration = time.time() - request_start
        
        response = outputs[0][inputs['input_ids'].shape[-1]:]
        
        results = tokenizer.decode(response, skip_special_tokens=True)
        #print(f"results: {results}")
        
        process_start = time.time()
        instruction_data = []
        raw_results = f"{num_prompt_instructions+1}. Instruction:" + results
        # 「######」で区切ってリストに分割
        raw_results_list = raw_results.split("######")
        #print(f"raw_results_list: {raw_results_list}")
        
        for i, result in enumerate(raw_results_list, start=1):
            print(f"{i} :result: {result}")
            if "指令：" in result or "輸入：" in result or "輸出：" in result:
                result = replace_keywords(result)
            if ("Instruction:" in result) and ("Input:" in result) and ("Output:" in result):
                new_instructions = post_process_response(result.strip())
                instruction_data += new_instructions
            
        print("----------------------------------------------------------")
        print("instruction_data: ")
        print(instruction_data)
        print("----------------------------------------------------------")
        
        total = len(instruction_data)
        print(f"total: {total}")
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenized instructions
            new_instruction = instruction_data_entry["instruction"]

            rouge_scores = rouge.get_scores([new_instruction] * len(all_instructions), all_instructions)
            rouge_scores = [score['rouge-l']['f'] for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                print(f"Filter: 最大rouge_scoresが0.7を超えています。")
                continue
            else:
                keep += 1
            
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            progress_bar.update(1)
            
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, generate_file_name))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)

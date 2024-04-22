import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from easyjailbreak.attacker.AutoDAN_Liu_2023 import *
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak import models


def main(
        model_name: str,
        template: str,
        limit: int = 100
        ):
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    # 加载model和tokenizer，作为target模型
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizers = AutoTokenizer.from_pretrained(model_name)
    llama2 = models.HuggingfaceModel(model,tokenizers, template)

    # your_key = ""
    # 加载attack model
    # openai_model = models.OpenaiModel(model_name='gpt-3.5-turbo', api_keys=your_key, template_name='chatgpt')

    # 加载数据
    dataset = JailbreakDataset('AdvBench')

    if limit:
        print(f"Limiting dataset to {limit} instances...")
        dataset._dataset = dataset._dataset[:limit]


    # attacker初始化
    attacker = AutoDAN(
        attack_model=llama2,
        target_model=llama2,
        jailbreak_datasets=dataset,
        eval_model = None,
        max_query=100,
        max_jailbreak=100,
        max_reject=100,
        max_iteration=100,
        device=device,
        num_steps=100,
        sentence_level_steps=5,
        word_dict_size=30,
        batch_size=4,
        num_elites=0.1,
        crossover_rate=0.5,
        mutation_rate=0.01,
        num_points=5,
        model_name=model_name,
        low_memory=1,
        pattern_dict=None
    )

    attacker.attack()

    output_path = f"outputs/AdvBench/{model_name}/AutoDAN.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    attacker.attack_results.save_to_jsonl(output_path)

if __name__ == "__main__":
    fire.Fire(main)
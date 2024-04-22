import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from easyjailbreak.attacker import GCG
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak import models


def main(
        model_name: str,
        template: str,
        limit: int = 30,
        device: str = "cuda:0"
        ):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizers = AutoTokenizer.from_pretrained(model_name)
    model = models.HuggingfaceModel(model,tokenizers, template)

    dataset = JailbreakDataset('AdvBench')

    if limit:
        print(f"Limiting dataset to {limit} instances...")
        dataset._dataset = dataset._dataset[:limit]

    attacker = GCG(
        attack_model=model,
        target_model=model,
        jailbreak_datasets=dataset,
        jailbreak_prompt_length=20,
        num_turb_sample=512,
        batchsize=1,      # decrease if OOM
        top_k=256,
        max_num_iter=100,
    )

    attacker.attack()

    output_path = f"data/AdvBench/{model_name}/GCG.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    attacker.attack_results.save_to_jsonl(output_path)

if __name__ == "__main__":
    fire.Fire(main)
import fire
import os
from tqdm.auto import tqdm
import jsonlines
from typing import Optional
from .run_jailbreak import get_defensed_model, get_model

def run_experiment(
    model_name: str,
    defense: Optional[str] = None,
    limit: Optional[int] = None,
    prompt_length: Optional[int] = 512,
    max_new_tokens: Optional[int] = 128,
):

    generation_config = dict(
        top_p=0.95,
        top_k=50,
        temperature=1.0,
        max_new_tokens=max_new_tokens,
    )

    print(f"Loading model {model_name}...")
    model, _ = get_model(model_name, prompt_length=prompt_length, generation_config=generation_config)

    with jsonlines.open(f"outputs/AdvBench/{model_name}/AutoDAN.jsonl") as f:
        dataset = list(f)
    
    if isinstance(defense, str) and defense != "":
        model = get_defensed_model(defense, model)
        print(f"Using defense {defense} on model")

    if defense:
        save_path = f"outputs/{model_name}/AutoDAN-{defense}.jsonl"
    else:
        save_path = f"outputs/{model_name}/AutoDAN.jsonl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with jsonlines.open(save_path, "w") as f:
        for item in tqdm(dataset, desc=f"{model_name} {defense} AutoDAN"):
            prompt = item["jailbreak_prompt"]
            prompt = prompt.format(query=item["query"])
            response = model.generate(prompt)
            item["response"] = response
            f.write(item)




if __name__ == "__main__":
    fire.Fire(run_experiment)
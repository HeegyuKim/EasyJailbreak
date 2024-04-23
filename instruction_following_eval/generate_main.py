import fire
from tqdm.auto import tqdm
from typing import Optional
import jsonlines
import os
from experiments.run_jailbreak import get_defensed_model, get_model



def run_experiment(
    model: str,
    defense: Optional[str] = None,
    prompt_length: Optional[int] = 1024,
    max_new_tokens: Optional[int] = 1024,
):

    generation_config = dict(
        top_p=0.95,
        top_k=50,
        temperature=1.0,
        max_new_tokens=max_new_tokens,
    )

    print(f"Loading target model {model}...")
    target_model, target_name = get_model(model, prompt_length=prompt_length, generation_config=generation_config)
    
    dataset = list(jsonlines.open("instruction_following_eval/data/input_data.jsonl"))
    
    if isinstance(defense, str):
        target_model = get_defensed_model(defense, target_model)
        print(f"Using defense {defense} on target model")
        save_path = f"outputs/{target_name}/IFEval-{defense}.jsonl"
    else:
        save_path = f"outputs/{target_name}/IFEval.jsonl"


    if os.path.exists(save_path):
        with jsonlines.open(save_path) as reader:
            skip_count = len(list(reader))
            dataset = dataset[skip_count:]
    else:
        skip_count = 0

    print(f"Saving results to {save_path}...")
    with jsonlines.open(save_path, "a") as writer:
        for i, instance in enumerate(tqdm(dataset, desc=f"IFEval-{target_name} {defense}")):
            output = target_model.generate(instance["prompt"], greedy=True)
            writer.write({"prompt": instance["prompt"], "response": output})


if __name__ == "__main__":
    fire.Fire(run_experiment)
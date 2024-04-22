import transformers

import fire
from tqdm.auto import tqdm
from typing import Optional
import jsonlines
import os

from experiments.run_jailbreak import get_model
from easyjailbreak.defense.self_refine import SelfRefineDefense, SelfRefineJSONDefense



def run_experiment(
    model: str,
    defense: str = "self-refine",
    target_attack: str = "Jailbroken",
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
    
    if defense == "self-refine":
        target_model = SelfRefineDefense(target_model)
    elif defense == "self-refine-json":
        target_model = SelfRefineJSONDefense(target_model)
    else:
        raise ValueError(f"Defense {defense} not recognized")


    print(f"Using defense {defense} on target model")
    input_file = f"outputs/{target_name}/{target_attack}.jsonl"
    save_path = f"outputs/{target_name}/{target_attack}-{defense}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        with jsonlines.open(save_path) as reader:
            skip_length = len(list(reader))
            print(f"Skipping {skip_length} instances")
    else:
        skip_length = 0

    dataset = list(jsonlines.open(input_file))
    dataset = dataset[skip_length:]

    print(f"Saving results to {save_path}...")
    with jsonlines.open(save_path, "a") as writer:
        for i, instance in enumerate(tqdm(dataset, desc=f"{target_attack}-{target_name} {defense}")):
            prompt = instance["jailbreak_prompt"].format(query=instance["query"])
            output, feedback = target_model.refine(prompt, instance["target_responses"][0])
            instance["original_response"] = instance["target_responses"][0]
            instance["response"] = output
            instance["feedback"] = feedback

            print(prompt)
            print("Feedback:", feedback)
            print("Refined:", output)

            writer.write(instance)


if __name__ == "__main__":
    fire.Fire(run_experiment)
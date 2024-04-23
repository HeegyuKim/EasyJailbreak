import fire
import os
from typing import Optional
from tqdm.auto import tqdm
import glob
import jsonlines
from easyjailbreak.models.flax_huggingface_model import FlaxAPI, FlaxHuggingfaceModel

# python -m experiments.eval_llamaguard

def main():
    generation_config = dict(
        top_p=0.95,
        top_k=50,
        temperature=1.0,
        max_new_tokens=4,
    )
    model = FlaxHuggingfaceModel(
        model_name_or_path='meta-llama/LlamaGuard-7b',
        prompt_length=2044,
        max_new_tokens=4,
        gen_args=generation_config,
        # fully_sharded_data_parallel=False,
        # mesh_axes_shape=(1, -1, 1, 1) if fsdp else (1, 1, 1, -1)
    )

    files = list(glob.glob("outputs/*/*/Jailbroken*.jsonl"))
    progress = tqdm(files, position=0, leave=True)
    for file in files:
        progress.set_description(file)

        filename = os.path.basename(file)
        output_file = f"{os.path.dirname(file)}/LlamaGuard/{filename}"


        if os.path.exists(output_file):
            with jsonlines.open(output_file) as reader:
                num_skip = len(list(reader))
                print(f"Skipping {num_skip} lines in {output_file}")
        else:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            num_skip = 0        

        with jsonlines.open(file) as reader:
            results = list(reader)[num_skip:]
        
        with jsonlines.open(output_file, 'a') as writer:
            for result in tqdm(results, position=1, leave=False):
                conv = [
                    {'role': 'user', 'content': result['query']},
                    {'role': 'assistant', 'content': result['target_responses'][0]}
                ]
                eval_result = model.chat(conv, greedy=True)
                print(conv)
                print("-->", eval_result)
                result['llamaguard'] = "unsafe" in eval_result

                writer.write(result)



if __name__ == "__main__":
    fire.Fire(main)
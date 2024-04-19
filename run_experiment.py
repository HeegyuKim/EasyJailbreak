import fire
from typing import Optional
from easyjailbreak.attacker import Jailbroken
from easyjailbreak.models.flax_huggingface_model import FlaxAPI, FlaxHuggingfaceModel
from easyjailbreak.datasets import JailbreakDataset


def get_model(name, prompt_length, generation_config):
    if "@" in name:
        model_type, model_name = name.split("@", 1)
    else:
        model_type = "flax"
        model_name = name

    if model_type == "flax":
        model = FlaxHuggingfaceModel(
            model_name_or_path=model_name,
            prompt_length=prompt_length,
            max_new_tokens=generation_config["max_new_tokens"],
            gen_args=generation_config,
        )
    elif model_type == "api":
        # api@0.0.0.0:35020/meta-llama/llama2
        server, model_name = model_name.split("/", 1)
        model = FlaxAPI(
            model_name=model_name,
            generation_config=generation_config,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, model_name

def run_experiment(
    target: str,
    attacker: str,
    eval: str,
    attack_model: Optional[str] = None,
    dataset: str = "AdvBench",
    num_attack: Optional[int] = None,
    prompt_length: Optional[int] = 512,
    max_new_tokens: Optional[int] = 128,
):

    generation_config = dict(
        top_p=0.95,
        top_k=50,
        temperature=1.0,
        max_new_tokens=max_new_tokens,
    )

    print(f"Loading target model {target}...")
    target_model, target_name = get_model(target, prompt_length=prompt_length, generation_config=generation_config)
    print(f"Loading eval model {eval}...")
    eval_model = get_model(eval, prompt_length=prompt_length, generation_config=generation_config)
    print(f"Loading dataset {dataset}...")
    dataset = JailbreakDataset(dataset)
    if num_attack:
        print(f"Limiting dataset to {num_attack} instances...")
        dataset._dataset = dataset._dataset[:num_attack]

    if attack_model is None:
        attack_model = target
        print(f"Using target model as attack model: {attack_model}")
    else:
        print(f"Loading attack model {attack_model}...")
        attack_model = get_model(attack_model, prompt_length=prompt_length, generation_config=generation_config)


    if attacker == "Jailbroken":
        attacker = Jailbroken(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset)
    else:
        raise ValueError(f"Unsupported attacker: {attacker}")
    
    attacker.attack()
    attacker.log()
    
    save_path = f"outputs/{target_name}/{attacker}.jsonl"
    attacker.attack_results.save_to_jsonl(save_path)



if __name__ == "__main__":
    fire.Fire(run_experiment)
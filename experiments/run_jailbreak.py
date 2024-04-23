import fire
import os
from typing import Optional
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from easyjailbreak.attacker import Jailbroken, CodeChameleon, PAIR, ReNeLLM, GPTFuzzer
try:
    from easyjailbreak.models.flax_huggingface_model import FlaxHuggingfaceModel
except ImportError:
    print("Flax is not installed. Flax models will not be available.")
    pass
from easyjailbreak.models.flax_api import FlaxAPI
from easyjailbreak.models.huggingface_model import HuggingfaceModel
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models import from_pretrained
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge, EvaluatorGenerativeJudgeFlaxLlamaGuard
from easyjailbreak.defense import SmoothLLMDefense, InContextDefense, SelfReminderDefense
from easyjailbreak.defense.self_refine import SelfRefineDefense



def get_model(name, prompt_length, generation_config):
    if "@" in name:
        model_type, model_name = name.split("@", 1)
    elif name == "GPTFuzz":
        model_type = "GPTFuzz"
        model_name = name
    else:
        model_type = "pt" if torch.cuda.is_available() else "flax"
        model_name = name

    if model_type == "flax":
        fsdp = "Starling-LM-7B" in model_name

        model = FlaxHuggingfaceModel(
            model_name_or_path=model_name,
            prompt_length=prompt_length,
            max_new_tokens=generation_config["max_new_tokens"],
            gen_args=generation_config,
            fully_sharded_data_parallel=False,
            mesh_axes_shape=(1, -1, 1, 1) if fsdp else (1, 1, 1, -1)
        )
    elif model_type == 'pt':
        if "Llama" in model_name:
            template = "llama-2"
        elif "Starling-LM" in model_name:
            template = "openchat_3.5"
        elif "zephyr" in model_name:
            template = "zephyr"
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model = from_pretrained(model_name, template, dtype=torch.bfloat16, **generation_config)
    elif model_type == "api":
        # api@0.0.0.0:35020/meta-llama/llama2
        server, model_name = model_name.split("/", 1)
        model = FlaxAPI(f"http://{server}")
    elif model_type == "openai":
        model = OpenaiModel(model_name=model_name, api_keys=os.environ['OPENAI_API_KEY'])
    elif model_type == "GPTFuzz":
        model_path = 'hubert233/GPTFuzz'
        judge_model = RobertaForSequenceClassification.from_pretrained(model_path).eval().cuda()
        judge_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = HuggingfaceModel(model=judge_model, tokenizer=judge_tokenizer, model_name='zero_shot', generation_config=generation_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, model_name

def get_defensed_model(defense, model):
    if defense == "self-reminder":
        defense_model = SelfReminderDefense(model)
    elif defense == "in-context":
        defense_model = InContextDefense(model)
    elif defense == "smooth-llm":
        defense_model = SmoothLLMDefense(model)
    elif defense == "self-refine":
        defense_model = SelfRefineDefense(model)
    else:
        raise ValueError(f"Unsupported defense: {defense}")
    return defense_model

def run_experiment(
    target: str,
    attacker: str,
    eval: str,
    defense: Optional[str] = None,
    eval_llama_guard: bool = True,
    attack_model: Optional[str] = None,
    dataset: str = "AdvBench",
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

    print(f"Loading target model {target}...")
    target_model, target_name = get_model(target, prompt_length=prompt_length, generation_config=generation_config)
    print(f"Loading eval model {eval}...")
    eval_model, _ = get_model(eval, prompt_length=prompt_length, generation_config=generation_config)
    print(f"Loading dataset {dataset}...")
    dataset = JailbreakDataset(dataset)
    if limit:
        print(f"Limiting dataset to {limit} instances...")
        dataset._dataset = dataset._dataset[:limit]
    
    if attack_model is None:
        attack_model = target_model
        print(f"Using target model as attack model: {target}")
    else:
        print(f"Loading attack model {attack_model}...")
        attack_model, _ = get_model(attack_model, prompt_length=prompt_length, generation_config=generation_config)
    
    if isinstance(defense, str) and defense != "":
        target_model = get_defensed_model(defense, target_model)
        print(f"Using defense {defense} on target model")

    attacker_name = attacker

    # llamaguard eval
    if attacker == "Jailbroken":
        attacker = Jailbroken(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset)
        if eval_llama_guard:
            attacker.evaluator = EvaluatorGenerativeJudgeFlaxLlamaGuard(eval_model)
    elif attacker == "ReNeLLM":
        attacker = ReNeLLM(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset)
    
    # LLM eval
    elif attacker == "PAIR":
        attacker = PAIR(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset)
    elif attacker == "CodeChameleon":
        attacker = CodeChameleon(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset)

    # Classifier Eval
    elif attacker == "GPTFuzz":
        attacker = GPTFuzzer(attack_model=attack_model,
                            target_model=target_model,
                            eval_model=eval_model,
                            jailbreak_datasets=dataset,
                            max_iteration=20,
                            max_query=  10000,
                            max_jailbreak= 1000,
                            max_reject= 10000)
    else:
        raise ValueError(f"Unsupported attacker: {attacker_name}")
    
    attacker.attack()
    attacker.log()
    
    if defense:
        save_path = f"outputs/{target_name}/{attacker_name}-{defense}.jsonl"
    else:
        save_path = f"outputs/{target_name}/{attacker_name}.jsonl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    attacker.attack_results.save_to_jsonl(save_path)



if __name__ == "__main__":
    fire.Fire(run_experiment)
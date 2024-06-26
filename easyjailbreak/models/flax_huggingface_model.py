import logging
import warnings
from .model_base import BlackBoxModelBase
from typing import Any, Dict, Optional, List, Union, Tuple

import functools
import copy
from tqdm.auto import tqdm
import requests
import transformers
from transformers import GenerationConfig

import jax, flax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from fjformer import get_dtype, make_shard_and_gather_fns, match_partition_rules, with_sharding_constraint

from .flax_partition_rules import get_partition_rules


MESH_SHAPES = {
    "fsdp": (1, -1, 1, 1),
    "tp": (1, 1, -1, 1),
    "sp": (1, 1, 1, -1),
}

class RNG:
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}

class FlaxHuggingfaceModel(BlackBoxModelBase):

    def __init__(
        self,
        model_name_or_path: str, 
        prompt_length: int,
        max_new_tokens: int,
        gen_args: Dict = {}, 
        fully_sharded_data_parallel = True,
        chat_template: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        mesh_axes_names = ("dp", "fsdp", "tp", "sp"),
        mesh_axes_shape: Union[Tuple, str] = (1, -1, 1, 1),
        dtype: str = "bf16",
    ):
        with jax.default_device(jax.devices("cpu")[0]):
            config = transformers.AutoConfig.from_pretrained(model_name_or_path)
            flax_model = transformers.FlaxAutoModelForCausalLM.from_config(
                config,
                _do_init=True,
                dtype=get_dtype(dtype),
                input_shape=(1, prompt_length + max_new_tokens)
                )

            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

            pt_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
            pt_state_dict = pt_model.state_dict()

            print("converting to flax")
            params = transformers.modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

            del pt_state_dict
            del pt_model
            import gc
            gc.collect()

            self.model = flax_model

        if isinstance(mesh_axes_shape, str):
            mesh_axes_shape = MESH_SHAPES[mesh_axes_shape]

        array = jnp.ones((len(jax.devices()), 1)).reshape(mesh_axes_shape)
        self.mesh = Mesh(mesh_utils.create_device_mesh(array.shape), mesh_axes_names)
        with self.mesh:
            logging.info(
                "matching partition rules"
            )
            partition_specs = match_partition_rules(params=params, rules=get_partition_rules(flax_model.config, fully_sharded_data_parallel=fully_sharded_data_parallel))
            shard_fns, _ = make_shard_and_gather_fns(partition_specs, get_dtype(dtype))
            logging.info(
                "sharding parameters across all of the chosen backend(tpu/gpu/cpu)s"
            )
            params = flax.traverse_util.flatten_dict(params)
            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
            pbar = tqdm(params.keys(), desc="Sharding Params")
            for key in pbar:
                key = tuple(key)
                params[key] = shard_fns[key](params[key])
            self.params = flax.traverse_util.unflatten_dict(params)
        self.partition_specs = partition_specs

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Setting pad token to eos token")
            
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        self.prefix_tokenizer = copy.deepcopy(tokenizer)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        self.tokenizer = copy.deepcopy(tokenizer)
        self.max_sequence_length = max_new_tokens + prompt_length
        self.prompt_length = prompt_length
        self.gen_args = gen_args
        self.rng_generator = RNG(42)

        generation_ps = PartitionSpec("dp", "fsdp")

        @functools.partial(
            pjit,
            in_shardings=(self.partition_specs, PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def greedy_generate(parameters, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, generation_ps)
            predict = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_length=self.max_sequence_length,

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,

                    temperature=self.gen_args.get("temperature"),
                    early_stopping=True,
                    do_sample=True,
                    num_beams=1,
                    top_p=self.gen_args.get("top_p"),
                    top_k=self.gen_args.get("top_k"),
                    repetition_penalty=self.gen_args.get("repetition_penalty")
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        @functools.partial(
            pjit,
            in_shardings=(self.partition_specs, PartitionSpec(), PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def generate(parameters, rng, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, generation_ps)
            predict = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                prng_key=rng,
                generation_config=GenerationConfig(
                    max_length=self.max_sequence_length,

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,

                    temperature=self.gen_args.get("temperature"),
                    early_stopping=True,
                    do_sample=True,
                    num_beams=1,
                    top_p=self.gen_args.get("top_p"),
                    top_k=self.gen_args.get("top_k"),
                    repetition_penalty=self.gen_args.get("repetition_penalty")
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        self.generate_function = generate
        self.greedy_generate_function = greedy_generate
        self._funcs_generated = True
        self.system_prompt = None

    def set_system_message(self, prompt):
        self.system_prompt = prompt
    
    def generate_prefix(self,
               string: str,
               greedy: bool = False,
               **kwargs
               ):
        fixed_pad = self.prompt_length
        tokens = self.prefix_tokenizer(
            string,
            max_length=fixed_pad,
            padding="max_length",
            truncation=True,
            return_tensors="jax"
        )

        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        with self.mesh:
            predicted_token = self.greedy_generate_function(
                self.params,
                input_ids,
                attention_mask
            ) if greedy else self.generate_function(
                self.params,
                self.rng_generator(),
                input_ids,
                attention_mask
            )

        return self.tokenizer.decode(predicted_token[0], skip_special_tokens=True)

    def generate(self,
               messages: str,
               generation_prefix: str = "",
               greedy: bool = False,
               clear_old_history=True,
               **kwargs
               ):
        if clear_old_history:
            self.conversation = []

        if isinstance(messages, str):
            messages = [messages]

        for index, message in enumerate(messages):
            self.conversation.append({
                'role': 'user' if index % 2 == 0 else 'assistant', 
                'content': message
                })
            

        return self.chat(self.conversation, generation_prefix, greedy, **kwargs)
        
    def chat(self,
               conversations: list,
               generation_prefix: str = "",
               greedy: bool = False,
               ):
        if conversations[0]['role'] != 'system' and self.system_prompt:
            conversations.insert(0, {'role': 'system', 'content': self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False) + generation_prefix
        return self.generate_prefix(prompt, greedy=greedy)

    def generate_messages(self, messages, clear_old_history=True, **kwargs):
        """
        Generates a response based on messages that include conversation history.
        :param list[str]|str messages: A list of messages or a single message string.
                                       User and assistant messages should alternate.
        :param bool clear_old_history: If True, clears the old conversation history before adding new messages.
        :return str: The response generated by the OpenAI model based on the conversation history.
        """
        if clear_old_history:
            self.conversation = []

        if isinstance(messages, str):
            messages = [messages]

        for index, message in enumerate(messages):
            self.conversation.append({
                'role': 'user' if index % 2 == 0 else 'assistant', 
                'content': message
                })
            
        response = self.forward_chat(self.conversation)
        return response

    def batch_generate(self, conversations, **kwargs):
        """
        Generates responses for multiple conversations in a batch.
        :param list[list[str]]|list[str] conversations: A list of conversations, each as a list of messages.
        :return list[str]: A list of responses for each conversation.
        """
        responses = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn('For batch generation based on several conversations, provide a list[str] for each conversation. '
                              'Using list[list[str]] will avoid this warning.')
            responses.append(self.generate_messages(conversation, **kwargs))
        return responses


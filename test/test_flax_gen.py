from easyjailbreak.models.flax_huggingface_model import FlaxHuggingfaceModel

if __name__ == "__main__":

    generator = FlaxHuggingfaceModel(
        'Felladrin/TinyMistral-248M-Chat-v2',
        prompt_length=512,
        max_new_tokens=512,
        fully_sharded_data_parallel=False,
        mesh_axes_shape = (1, 1, 1, -1),
    )

    # output = generator.generate("Hello, my name is")
    # print(output)

    output = generator.chat([{
        'role': 'user',
        'content': "Hello, my name is Kim. How are you today?"
        }])
    
    print(output)
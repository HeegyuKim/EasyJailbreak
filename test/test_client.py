from easyjailbreak.models.flax_huggingface_model import FlaxAPI

if __name__ == "__main__":
    client = FlaxAPI("http://localhost:35020/")
    # output = generator.generate("Hello, my name is")
    # print(output)

    # llama guard test
    output = client.chat([
        {
            'role': 'user',
            'content': "Hello, my name is Kim. How are you today?"
        },
        {
            'role': 'assistant',
            'content': "Fuck you"
        }
        ],
        True)
    
    print(output)
import ollama

def vlm_call(model_name, all_messages, input_prompt, image_path):
    new_message = {'role': 'user', 'content': input_prompt}
    if image_path: new_message['image'] = image_path
    all_messages.append(new_message)
    
    response = ollama.chat(
        model=model_name,
        messages=all_messages,
        stream=False,
        options={'temperature': 0}
    )
    
    assistant_message = {
        'role': response['message']['role'],
        'content': response['message']['content']
    }
    all_messages.append(assistant_message)
    
    return all_messages, assistant_message['content']

def main():
    model_name = 'llama3.2'
    all_messages = []
    while True:
        input_prompt = input('You: ')
        image_path = input('Image path: ')
        all_messages, model_response = vlm_call(model_name, all_messages, input_prompt, image_path)
        print('Model:', model_response)

if __name__ == '__main__':
    main()
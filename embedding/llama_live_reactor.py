import os
from fastrtc import (AdditionalOutputs, ReplyOnPause, Stream, get_stt_model, get_tts_model)
import ollama
import cv2
import time
import gradio as gr

models = [model.model for model in ollama.list().models]
print(f"Available models: {models}")

if len(models) == 0:
    print("Error: No models available.")
    exit()

find_llama = -1
for ind in range(len(models)):
    if models[ind] == "llama3.2-vision:11b":
        find_llama = ind
        break

if find_llama == -1:
    print(f"llama3.2-vision not installed, using first available model")
    selected_model = models[0]
else:
    selected_model = models[find_llama]
print(f"Using model: {selected_model}")

stt_model = get_stt_model()
tts_model = get_tts_model()

MESSAGES_TEMP = []

def echo(audio, gradio_chatbot, markdown):
    global MESSAGES_TEMP
    print(gradio_chatbot)
    if len(gradio_chatbot) == 0:
        MESSAGES_TEMP = []
    prompt = stt_model.stt(audio) 

    picture_demand = prompt.lower().find("take a picture")
    if picture_demand != -1:
        prompt = str(prompt[0:picture_demand] + "look at this picture" + prompt[picture_demand + len("take a picture"):])
        print(f"User: {prompt}\n")

        gradio_chatbot.append(gr.ChatMessage(role= 'user', content= "Take a picture."))
        yield AdditionalOutputs(gradio_chatbot)

        gradio_chatbot.append(gr.ChatMessage(role= 'assistant', content= "Okay, taking a picture."))
        yield AdditionalOutputs(gradio_chatbot)

        for audio_chunk in tts_model.stream_tts_sync("Okay, taking a picture."):
            yield audio_chunk

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        timestr = time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime()) 
        cv2.imwrite(f"capture_{timestr}.jpg", frame)

        print("Picture taken\n")

        MESSAGES_TEMP.append({
            'role': 'user',
            'content': prompt,
            'images': [f"capture_{timestr}.jpg"],
        })

        gradio_chatbot.append(gr.ChatMessage(role= 'assistant', content= gr.Image(f"capture_{timestr}.jpg")))
        yield AdditionalOutputs(gradio_chatbot)
        
        gradio_chatbot.append(gr.ChatMessage(role= 'assistant', content= "Picture taken, looking at it now."))
        yield AdditionalOutputs(gradio_chatbot)

        for audio_chunk in tts_model.stream_tts_sync("Picture taken, looking at it now."):
            yield audio_chunk

    else:
        print(f"User: {prompt}\n")
        MESSAGES_TEMP.append({
            'role': 'user',
            'content': prompt,
        })
    
    gradio_chatbot.append(gr.ChatMessage(role= 'user', content= prompt))
    yield AdditionalOutputs(gradio_chatbot)

    chatbot = ollama.chat(
                model=selected_model,
                messages=[*MESSAGES_TEMP],
            )

    response = chatbot["message"]["content"]
    print(f"{selected_model}: {response}\n")

    MESSAGES_TEMP.append({
        'role': 'assistant',
        'content': response,
    })

    gradio_chatbot.append(gr.ChatMessage(role= 'assistant', content= response))
    yield AdditionalOutputs(gradio_chatbot)

    for audio_chunk in tts_model.stream_tts_sync(response):
        yield audio_chunk


chatbot = gr.Chatbot(type="messages", label=f"{selected_model}", resizable=True, autoscroll=True, height=600)
markdown = gr.Markdown(f"Say 'Take a picture' to take a picture and show it to the chatbot")
stream = Stream(ReplyOnPause(echo), 
                modality="audio", 
                mode="send-receive", 
                additional_outputs_handler=lambda a, b: b,
                additional_inputs=[chatbot, markdown],
                additional_outputs=[chatbot],
                ui_args={
                    "title": f"Multimodal Chatbot with {selected_model}",
                    },
                )

#stream.ui.launch(share=False)
stream.ui.launch(share=True)
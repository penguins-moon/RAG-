import gradio as gr
from retrieve_documents import knowledge
import os
import sys
import pandas as pd

os.chdir(sys.path[0])
Base = knowledge('./knowledgebase.json')

# Gradio界面
def slow_echo(prompt, history):
    print(prompt)
    message = Base.generate(prompt)
    # message = ask_glm(api_key,prompt)
    for i in range(len(message)):
        yield message[: i+1]


with gr.Blocks(theme='soft',title='公司年报助手') as app:
    gr.Markdown("# 公司年报助手")
    with gr.Row():
        with gr.Column(scale=0.4):
            # 文件传入接口
                file_input = gr.File(label="Upload Your PDF File Here...", file_count='single')
                file_list = gr.Dataframe(value=pd.DataFrame(Base.get_keys(), columns=["Available data"]), interactive=False, show_label=False)
                file_input.upload(fn = Base.upload_files, inputs=file_input, outputs=file_list)       
        
        with gr.Column(scale=0.6):
            chatbot = gr.ChatInterface(slow_echo, title="Chatbot GLM")

app.launch(inbrowser=True, share=True)
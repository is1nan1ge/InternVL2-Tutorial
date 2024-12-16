import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from utils import load_json, init_logger
from demo import ConversationalAgent, CustomTheme

FOOD_EXAMPLES = "demo/food_for_demo.json"
MODEL_PATH = "/root/share/new_models/OpenGVLab/InternVL2-2B"
# MODEL_PATH = "/root/xtuner/work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/lr35_ep10"
OUTPUT_PATH = "./outputs"

def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    setup_seeds()
    # logging
    init_logger(OUTPUT_PATH)
    # food examples
    food_examples = load_json(FOOD_EXAMPLES)
    
    agent = ConversationalAgent(model_path=MODEL_PATH,
                                outputs_dir=OUTPUT_PATH)
    
    theme = CustomTheme()
    
    titles = [
        """<center><B><font face="Comic Sans MS" size=10>趣味美食小助手</font></B></center>"""  ## Kalam:wght@700
        """<center><B><font face="Courier" size=5>InternVL 多模态模型部署微调实践</font></B></center>"""
    ]
    
    language = """Language: 中文 and English"""
    with gr.Blocks(theme) as demo_chatbot:
        for title in titles:
            gr.Markdown(title)
        # gr.Markdown(article)
        gr.Markdown(language)
        
        with gr.Row():
            with gr.Column(scale=3):
                start_btn = gr.Button("Start Chat", variant="primary", interactive=True)
                clear_btn = gr.Button("Clear Context", interactive=False)
                image = gr.Image(type="pil", interactive=False)
                upload_btn = gr.Button("🖼️ Upload Image", interactive=False)
                
                with gr.Accordion("Generation Settings"):                    
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.1,
                                      value=0.8,
                                      interactive=True,
                                      label='top-p value',
                                      visible=True)
                    
                    temperature = gr.Slider(minimum=0, maximum=1.5, step=0.1,
                                            value=0.8,
                                            interactive=True,
                                            label='temperature',
                                            visible=True)
                    
            with gr.Column(scale=7):
                chat_state = gr.State()
                chatbot = gr.Chatbot(label='InternVL2', height=800, avatar_images=((os.path.join(os.path.dirname(__file__), 'demo/user.png')), (os.path.join(os.path.dirname(__file__), "demo/bot.png"))))
                text_input = gr.Textbox(label='User', placeholder="Please click the <Start Chat> button to start chat!", interactive=False)
                gr.Markdown("### 输入示例")
                def on_text_change(text):
                    return gr.update(interactive=True)
                text_input.change(fn=on_text_change, inputs=text_input, outputs=text_input)
                gr.Examples(
                    examples=[["图片中的食物属于哪个菜系?"],
                              ["如果让你用简单的语言形容一下品尝图片中的食物的味道，你会怎么说?"],
                              ["去哪个地方游玩时应该品尝当地的特色美食图片中的食物?"],
                              ["食用图片中的食物时，一般它上菜或摆盘时的特点是?"]],
                    inputs=[text_input]
                )
        
        with gr.Row():
            gr.Markdown("### 食物快捷栏")
        with gr.Row():
            example_xinjiang_food = gr.Examples(examples=food_examples["新疆菜"], inputs=image, label="新疆菜")
            example_sichuan_food = gr.Examples(examples=food_examples["川菜（四川，重庆）"], inputs=image, label="川菜（四川，重庆）")
            example_xibei_food = gr.Examples(examples=food_examples["西北菜 （陕西，甘肃等地）"], inputs=image, label="西北菜 （陕西，甘肃等地）")
        with gr.Row():
            example_guizhou_food = gr.Examples(examples=food_examples["黔菜 (贵州）"], inputs=image, label="黔菜 (贵州）")
            example_jiangsu_food = gr.Examples(examples=food_examples["苏菜（江苏）"], inputs=image, label="苏菜（江苏）")
            example_guangdong_food = gr.Examples(examples=food_examples["粤菜（广东等地）"], inputs=image, label="粤菜（广东等地）")
        with gr.Row():
            example_hunan_food = gr.Examples(examples=food_examples["湘菜（湖南）"], inputs=image, label="湘菜（湖南）")
            example_fujian_food = gr.Examples(examples=food_examples["闽菜（福建）"], inputs=image, label="闽菜（福建）")
            example_zhejiang_food = gr.Examples(examples=food_examples["浙菜（浙江）"], inputs=image, label="浙菜（浙江）")
        with gr.Row():
            example_dongbei_food = gr.Examples(examples=food_examples["东北菜 （黑龙江等地）"], inputs=image, label="东北菜 （黑龙江等地）")
            
                
        start_btn.click(agent.start_chat, [chat_state], [text_input, start_btn, clear_btn, image, upload_btn, chat_state])
        clear_btn.click(agent.restart_chat, [chat_state], [chatbot, text_input, start_btn, clear_btn, image, upload_btn, chat_state], queue=False)
        upload_btn.click(agent.upload_image, [image, chatbot, chat_state], [image, chatbot, chat_state])
        text_input.submit(
            agent.respond,
            inputs=[text_input, image, chatbot, top_p, temperature, chat_state], 
            outputs=[text_input, image, chatbot, chat_state]
        )

    demo_chatbot.launch(share=True, server_name="127.0.0.1", server_port=1096, allowed_paths=['./'])
    demo_chatbot.queue()
    

if __name__ == "__main__":
    main()

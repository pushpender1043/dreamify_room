import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import random
import os

# ====== Load GPT-2 Designer (D-Zen) ======
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ====== Load Stable Diffusion ======
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# ====== Simulated Accuracy Estimator ======
def estimate_accuracy(prompt, image):
    score = random.uniform(0.7, 0.95)  # Simulated score
    return f"🎯 Estimated Design Match Score: {round(score * 100, 2)}%"

# ====== Generate Room Prompt from GPT-2 ======
def generate_design_prompt(user_desc):
    character_intro = "You are D-Zen, a Zen-inspired AI interior designer with futuristic taste and mindfulness."
    prompt = f"{character_intro} A client wants the following room: '{user_desc}'. Describe in vivid detail the design you'd create:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=200,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    description = decoded.split("the design you'd create:")[-1].strip()
    return description

# ====== Full Generation Function ======
def dreamify_room(user_input):
    gpt_prompt = generate_design_prompt(user_input)
    image = pipe(gpt_prompt).images[0]
    score = estimate_accuracy(gpt_prompt, image)
    return gpt_prompt, image, score

# ====== Gradio UI ======
with gr.Blocks(theme=gr.themes.Soft(), css="body {background-color: #f0f8ff}") as app:
    gr.Markdown("""
    # 🛋️ **Dreamify by D-Zen**
    _Your Zen-inspired Dream Room Generator powered by GPT-2 & Stable Diffusion_

    ![logo](https://img.icons8.com/emoji/96/bed-emoji.png)
    """)

    with gr.Row():
        user_input = gr.Textbox(label="🧠 Describe Your Dream Room", placeholder="e.g. A minimalist reading nook with forest view")
        generate_btn = gr.Button("🎨 Dreamify Room")

    with gr.Row():
        prompt_out = gr.Textbox(label="🧾 D-Zen's Design Prompt")
        image_out = gr.Image(label="🖼️ Generated Room Image")
        score_out = gr.Textbox(label="📊 Design Accuracy Score")

    generate_btn.click(fn=dreamify_room, inputs=user_input, outputs=[prompt_out, image_out, score_out])

app.launch()

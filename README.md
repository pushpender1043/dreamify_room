# 🛋️ Dreamify by D-Zen: AI Interior Designer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers%20%26%20Diffusers-yellow)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-orange)

> **"Where Zen philosophy meets Generative AI to design your dream space."**

**Dreamify** is a Multimodal Generative AI application that acts as your personal interior designer. It uses a chained-model approach: **GPT-2** functions as a creative director ("D-Zen") to brainstorm detailed design concepts, which are then visualized into photorealistic images using **Stable Diffusion**.

---

## 🖼️ Project Demo

![Dreamify Interface Screenshot](path/to/your/screenshot.png)
*(Note: Replace this line with a screenshot of your Gradio interface running)*

---

## 🚀 Key Features

* **🤖 Multimodal Pipeline:** Seamlessly connects NLP (Text generation) with Computer Vision (Image generation).
* **🧠 Intelligent Prompt Expansion:** Uses **GPT-2** to take simple user inputs (e.g., "a blue bedroom") and expands them into vivid, artistic descriptions suited for image synthesis.
* **🎨 High-Fidelity Visualization:** Utilizes **Stable Diffusion v1.5** to generate realistic interior renderings.
* **🧘 D-Zen Persona:** The AI is prompted to act as a "Zen-inspired designer," ensuring aesthetically pleasing and mindful outputs.
* **📊 Accuracy Estimator:** Includes a simulated logic module to provide a "Design Match Score" for user engagement.
* **💻 Interactive UI:** Built with **Gradio** for an easy-to-use, browser-based interface.

---

## 🛠️ Tech Stack

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Language Model** | GPT-2 (via Transformers) | Creative Prompt Engineering & Text Generation |
| **Image Model** | Stable Diffusion v1.5 (via Diffusers) | Text-to-Image Synthesis |
| **Deep Learning** | PyTorch | Tensor operations & GPU acceleration |
| **Interface** | Gradio | Frontend UI creation |
| **Environment** | Google Colab (Recommended) | GPU-based execution |

---

## ⚙️ How It Works (The Logic)

1.  **Input:** User describes a room (e.g., *"A futuristic gaming room"*).
2.  **Processing (NLP):**
    * The input is fed into **GPT-2**.
    * GPT-2 (conditioned as "D-Zen") rewrites the prompt, adding details about lighting, textures, and atmosphere.
3.  **Visualization (CV):**
    * The enhanced prompt is passed to the **Stable Diffusion Pipeline**.
    * The model runs inference (preferably on CUDA) to generate the image.
4.  **Output:** The final image, the enhanced text prompt, and a confidence score are displayed on the Gradio dashboard.

---

## 📦 Installation & Usage

Since this project requires GPU power, it is recommended to run it on **Google Colab** or a machine with a dedicated NVIDIA GPU.

### Prerequisites
```bash
pip install torch transformers diffusers gradio accelerate

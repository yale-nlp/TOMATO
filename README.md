# 🍅 TOMATO


## 1. Proprietary model evaluation 


### 1.1 Install all necessary libraries
You need to create a conda virtual environment first: 
```bash
conda create -n tomato 
conda activate tomato
```
Then, install dependencies:
```bash 
pip install openai # GPT 
pip install google-generativeai # Gemini 
pip install anthropic # claude
pip install reka-api==2.0.0 # reka
```

### 1.2 Run the inference
```bash
git clone https://github.com/yale-nlp/TOMATO.git
cd TOMATO
python src/evaluate.py --model gpt-4o-mini --total_frames 4
```
Models can be selected from:
- GPT: `["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]`
- Claude 3 & 3.5: `["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620"]`
- Gemini: `["gemini-1.5-flash", "gemini-1.5-pro"]`
- Reka: `["reka-core-20240501", "reka-flash-20240226", "reka-edge-20240208"]`



## 2. Open-source model evaluation 

### 2.1 InternVL-2
First, download the checkpoint of 3 models:

```bash 
# Go the the root directory
mkdir pretrained && cd pretrained 
# Download OpenGVLab/InternVL2-26B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-26B --local-dir InternVL2-26B
# Download OpenGVLab/InternVL2-40B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-40B --local-dir InternVL2-40B
# Download OpenGVLab/InternVL2-Llama3-76B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-Llama3-76B --local-dir InternVL2-Llama3-76B
```

Run the inference. Currently there's a bug in the `src/evaluate.py` function and I only use 1 GPU and set `world_size = 1` in line 94 in `src/generate_lib/internvl.py`. We can run 26B and 40B model with 1 80GB A100 GPU; but we need 2 A100 GPUs for 76B (I have a bug saying `you have 2 devices, cuda:0 and cuda:1`).
```bash
# InternVL2-26B
python src/evaluate.py --model InternVL2-26B --total_frames 8
# InternVL2-40B
python src/evaluate.py --model InternVL2-40B --total_frames 8
# InternVL2-Llama3-76B
# Need to be fixed
python src/evaluate.py --model InternVL2-Llama3-76B --total_frames 8
```

### 2.2 QQMM: Video-CCAM-14B
Install some packages:
```bash
pip install torch==2.1.0 \
torchvision==0.16.0 \
transformers==4.40.2 \
peft==0.10.0 \
pyarrow==13.0.0 \
decord==0.6.0 \
pysubs2==1.7.2
opencv-python \
imageio \
pandas \
flash_attn==2.5.8 \
accelerate==0.31.0
```

You need to clone the `Video-CCAM` repo:
```bash
cd src/generate_lib
git clone https://github.com/QQ-MM/Video-CCAM.git
```
Need to change the directory name from `Video-CCAM` to `Video_CCAM`.

Then, you need to download the vision encoder and llm into `./pretrained directory` for running 14B model:

```bash
cd pretrained
# video-ccam-14B
huggingface-cli download --resume-download --local-dir-use-symlinks False JaronTHU/Video-CCAM-14B-v1.1 --local-dir Video-CCAM-14B-v1.1
# Phi-3-medium
huggingface-cli download --resume-download --local-dir-use-symlinks False microsoft/Phi-3-medium-4k-instruct --local-dir Phi-3-medium-4k-instruct
# vision encoder
huggingface-cli download --resume-download --local-dir-use-symlinks False google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
```

Run `9B` model:
```bash
# 9B
huggingface-cli download --resume-download --local-dir-use-symlinks False JaronTHU/Video-CCAM-9B-v1.1 --local-dir Video-CCAM-9B-v1.1
# Yi-9.5B
huggingface-cli download --resume-download --local-dir-use-symlinks False 01-ai/Yi-1.5-9B-Chat --local-dir Yi-1.5-9B-Chat
```

Run `4B` model:
```bash
# 4B
huggingface-cli download --resume-download --local-dir-use-symlinks False JaronTHU/Video-CCAM-4B-v1.1 --local-dir Video-CCAM-4B-v1.1
# Phi-3-mini
huggingface-cli download --resume-download --local-dir-use-symlinks False microsoft/Phi-3-mini-4k-instruct --local-dir Phi-3-mini-4k-instruct
```



### 2.3 LLaVA One Vision and LLaVA-NeXT

You need to clone the LLaVA-NeXT repo inside `generate_lib` directory:
```
cd src/generate_lib
git clone git@github.com:LLaVA-VL/LLaVA-NeXT.git
```
Then, change the directory name to `LLaVA_NeXT`.
 
Download model checkpoint:

```bash
cd pretrained
# LLaVA-NeXT-Video-32B-Qwen
huggingface-cli download --resume-download --local-dir-use-symlinks False lmms-lab/LLaVA-NeXT-Video-32B-Qwen --local-dir LLaVA-NeXT-Video-32B-Qwen
# LLaVA-One-Vision-Qwen-2-7B
huggingface-cli download --resume-download --local-dir-use-symlinks False lmms-lab/llava-onevision-qwen2-7b-ov --local-dir llava-onevision-qwen2-7b-ov
# LLaVA-One-Vision-Qwen-2-72B
huggingface-cli download --resume-download --local-dir-use-symlinks False lmms-lab/llava-onevision-qwen2-72b-ov --local-dir llava-onevision-qwen2-72b-ov
```

Might want to create a new virtual environment:
```
cd src/generate_lib/LLaVA_NeXT
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

Check `src/evaluate.py` for inference. 

### 2.4 Intern Video 2

```bash
# OpenGVLab/InternVideo2-Chat-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVideo2-Chat-8B --local-dir InternVideo2-Chat-8B
```

```
python src/evaluate.py --model InternVideo2-Chat-8B --total_frames 10 --reasoning_type rotation
```

### 2.5 VideoLLaMA2

```bash
# video LLaMA 2 72B
huggingface-cli download --resume-download --local-dir-use-symlinks False DAMO-NLP-SG/VideoLLaMA2-72B --local-dir VideoLLaMA2-72B

# video LLaMA 2 7B
huggingface-cli download --resume-download --local-dir-use-symlinks False DAMO-NLP-SG/VideoLLaMA2-7B --local-dir VideoLLaMA2-7B
```


```bash
python src/evaluate.py --model VideoLLaMA2-72B --reasoning_type rotation_composite --total_frames 16
```


### 2.6 QwenVL 2

```bash
# 72B
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2-VL-72B-Instruct --local-dir Qwen2-VL-72B-Instruct
# 72B-AWQ 
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2-VL-72B-Instruct-AWQ --local-dir Qwen2-VL-72B-Instruct-AWQ
# 7B
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2-VL-7B-Instruct --local-dir Qwen2-VL-7B-Instruct

```

# Citation
If you find 🍅TOMATO useful for your research and applications, please cite using this BibTex:

```bibtex
@misc{shangguan2024tomatoassessingvisualtemporal,
      title={TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models}, 
      author={Ziyao Shangguan and Chuhan Li and Yuxuan Ding and Yanan Zheng and Yilun Zhao and Tesca Fitzgerald and Arman Cohan},
      year={2024},
      eprint={2410.23266},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.23266}, 
}
```
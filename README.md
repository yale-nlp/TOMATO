# 🍅 TOMATO

#### [**📄 Paper**](https://arxiv.org/abs/2410.23266) | [**🤗 Data**](https://huggingface.co/datasets/yale-nlp/TOMATO) | [**🎬 Videos**](https://drive.google.com/file/d/1-dNt9bZcp6C3RXuGoAO3EBgWkAHg8NWR/view?usp=drive_link) 

This repository contains the implementation of the following paper:

>🍅 TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models <br>
>[Ziyao Shangguan](https://ziyaosg.github.io/)\*<sup>1</sup>,&nbsp;
[Chuhan Li](https://LeeChuh.github.io)\*<sup>1</sup>,&nbsp;
[Yuxuan Ding](https://scholar.google.com/citations?user=jdsf4z4AAAAJ)<sup>1</sup>,&nbsp;
[Yanan Zheng](https://scholar.google.com/citations?user=0DqJ8eIAAAAJ)<sup>1</sup>,&nbsp;
[Yilun Zhao](https://yilunzhao.github.io/)<sup>1</sup>,&nbsp;
[Tesca Fitzgerald](https://www.tescafitzgerald.com/)<sup>1</sup>,&nbsp;
[Arman Cohan](https://armancohan.com/)<sup>1</sup><sup>2</sup> <br>
>*Equal contribution. <br>
><sup>1</sup>Yale University &nbsp;<sup>2</sup>Allen Institute of AI <sup>


## TOMATO - A Visual Temporal Reasoning Benchmark
![figure1](/misc/figure1.png)

### Introduction

Our study of existing benchmarks shows that visual temporal reasoning capabilities of Multimodal Foundation Models (MFMs) are likely overestimated as many questions can be solved by using a single, few, or out-of-order frames. To systematically examine current visual temporal reasoning tasks, we propose three principles with corresponding metrics: (1) *Multi-Frame Gain*, (2) *Frame Order Sensitivity*, and (3) *Frame Information Disparity*. 

Following these principles, we introduce TOMATO, a novel benchmark crafted to rigorously assess MFMs' temporal reasoning capabilities in video understanding. TOMATO comprises 1,484 carefully curated, human-annotated questions spanning 6 tasks (i.e. *action count*, *direction*, *rotation*, *shape&trend*, *velocity&frequency*, and *visual cues*), applied to 1,417 videos, including 805 self-recorded and -generated videos, that encompass 3 video scenarios (i.e. *human-centric*, *real-world*, and *simulated*). In the 805 self-created videos, we apply editing to incorporate *counterfactual scenes*, *composite motions*, and *zoomed-in* views, aiming to investigate the impact of these characteristics on the performance of MFMs.

### Task Examples

![rotation](/misc/ball_rotation_frames.png)
>What direction(s) does the Ping Pong ball rotate in? <br>
>A. Clockwise throughout. <br>
>B. No rotation. <br>
>C. Clockwise then counter-clockwise. <br>
>D. Counter-clockwise throughout. <br>
>E. Counter-clockwise then clockwise. <br>
>
>Answer: D. Counter-clockwise throughout. <br>

![acceleration](/misc/dropping_reversed_frames.png)
>What is the pattern of the object’s speed in the video? <br>
>A. Not moving at all. <br>
>B. Constant speed. <br>
>C. Decelerating. <br>
>D. Accelerating. <br>
>
>Answer: C. Decelerating.


![human_gesture](/misc/human_gesture_frames.png) <br>
>What instruction did the person give to the camera in the video? <br>
>A. Moving Down. <br>
>B. Moving Left. <br>
>C. Moving Further. <br>
>D. Moving Closer. <br>
>E. Moving Right. <br>
>F. Moving Up. <br>
>
>Answer: E. Moving Right.


![synthetic_human](/misc/synthetic_human_frames.png) <br>
>How many triangle(s) does the person draw in the air throughout the entire video? <br>
>A. 0 <br>
>B. 1 <br>
>C. 2 <br>
>D. 3 <br>
>E. 4 <br>
>F. 5 <br>
>
>Answer: E. 4

### Analysis Highlight

![earth_moon_frames](/misc/earth_moon_frames.png)

Our in-depth error case analysis reveals that **models lack the basic ability to interpret frames as a continuous sequence**. In the example, while GPT-4o correctly generates captions for each consecutive change in the moon's movement, showcasing its ability to reason at individual time steps, it still fails to infer based on the captions that the overall sequence represents a clockwise rotation and falsely concludes that it is a counter-clockwise rotation. 

For more detailed error case analysis, please refer to Section 6.3 in our paper.


## Dataset and Evaluation
### 1. Setup 

```bash
git clone https://github.com/yale-nlp/TOMATO
cd TOMATO
```
Download the [videos](https://drive.google.com/file/d/1-dNt9bZcp6C3RXuGoAO3EBgWkAHg8NWR/view?usp=drive_link) and unzip into the /TOMATO directory

<details>
<summary>After downloading the videos, your file structure should look like this.</summary>

```
.
├── data/
├── src/
├── videos/
│   ├── human/
│   ├── object/
│   ├── simulated/

```
</details>


#### 1.1 Proprietary Models 
To install the required packages for evaluating proprietary models, run:
```bash
pip install openai # GPT 
pip install google-generativeai # Gemini 
pip install anthropic # Claude
pip install reka-api==2.0.0 # Reka
```
Create a `.env` file in the root directory with the following format:
```
OPENAI_API_KEY="your_openai_api_key"
GEMINI_API_KEY="your_gemini_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
REKA_API_KEY="your_reka_api_key"
```

#### 1.2 Open-sourced Models
Create a directory named `pretrained` in the root of TOMATO to store open-sourced models. For example, to download `Qwen-2-VL-7B` model, run the following command: 

```bash
mkdir pretrained && cd pretrained
huggingface-cli download 
  --resume-download 
  --local-dir-use-symlinks False Qwen/Qwen2-VL-7B-Instruct 
  --local-dir Qwen2-VL-7B-Instruct
```

<details>
  <summary>After downloading open-sourced models, your file structure should look like this.</summary>

```
.
├── data/
├── src/
├── videos/
├── pretrained/
│   ├── Qwen2-VL-7B-Instruct/
│   ├── ...
```
</details>
<br>

**Note**: To use `Video-CCAM`, `LLaVA-NeXT`, `Video-LLaVA`, `VideoLLaMA2`,  and `VILA`, follow additional instructions below. <br>
Clone their repositories into the `./src/generate_lib/` directory. Run the following commands:
```bash
cd ./src/generate_lib

git clone git@github.com:QQ-MM/Video-CCAM.git             # Video-CCAM
git clone git@github.com:LLaVA-VL/LLaVA-NeXT.git          # LLaVA-NeXT
git clone git@github.com:DAMO-NLP-SG/VideoLLaMA2.git      # VideoLLaMA2
git clone git@github.com:PKU-YuanGroup/Video-LLaVA.git    # Video-LLaVA
git clone git@github.com:NVlabs/VILA.git                  # VILA
```
After cloning, rename the directories by replacing hyphens (`-`) with underscores (`_`):
```bash
mv Video-CCAM Video_CCAM
mv LLaVA-NeXT LLaVA_NeXT
mv Video-LLaVA Video_LLaVA
```

### 2. Evaluation

To run evaluation with a model:
```bash
python src/evaluate.py 
  --model $model_name
  --reasoning_type ALL 
  --demonstration_type ALL 
  --total_frames $total_frames
```
All supported models are listed [here](https://github.com/yale-nlp/TOMATO/blob/2161ce9a98291ce4fcb7aff9a531d10502bf5b10/src/config.json#L2-L62). To evaluate additional models, please refer to the next section.<br>

[This](https://github.com/yale-nlp/TOMATO/blob/2161ce9a98291ce4fcb7aff9a531d10502bf5b10/src/config.json#L63-L70) is a list of models that take in videos directly and any specified `total_frames` will be ignore. <br>

You can specify a subset of `reasoning_type` and `demonstration_type` using a comma-seperated list. [These](https://github.com/yale-nlp/TOMATO/blob/2161ce9a98291ce4fcb7aff9a531d10502bf5b10/src/config.json#L71-83) are the lists of valid choices.

### 3. Result Parsing
When our standard parser using regular expression fails, we employ `GPT-4o-mini` to extract answers from model response. To use the parser:
```bash
python src/parse_result.py
``` 
**Note**: This parser is designed to be incremental. It only parses additional raw model responses while leaving the already parsed results unchanged.

### 4. Display Categorized Scores

Scores are grouped by `model`, `reasoning_type`+`model`, and `demonstration_type`+`model`. To display scores:

```bash
python src/get_categorized_score.py
```

## Evaluate Additional Models

Our evaluation scripts are designed for the ease of adding additional models, simply:

### 1. Add Model to Config File
Add `model_family` and `model_name` to `src/config.json` like below:

```json
{
    "models": {
        "{model_family}": [
            "{model_name}",
            "..."
        ]
```

### 2. Add Model Evaluation Code
Create the corresponding `{model_family}.py` file under `src/generate_lib` with the starter code below:

```python
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS, GENERATION_SEED
from generate_lib.construct_prompt import construct_prompt
from generate_lib.utils import read_video

def generate_response(model_name: str, queries: list, total_frames: int, output_dir: str):
    # initialize your model 
    model = ...

    for query in queries:
      id_ = query['id']
      question = query['question']
      gt = optionized_list[query['answer']]

      # construct prompt
      base64Frames, _ = read_video(video_path=video_path, total_frames=total_frames)
      prompt, all_choices, index2ans = construct_prompt(question=question,
                                                        options=options,
                                                        num_frames=total_frames)
      
      # generate response
      response = model(...)

      # save model response in default format to use our result parser
      with open(output_dir, "a") as f:
            f.write(json.dumps(
                {
                    "id": id_,
                    "question": question,
                    "response": response,
                    "all_choices": all_choices,
                    "index2ans": index2ans,
                    'gt': gt
                }
            ) + "\n")
```


## Experiments

### 1. Comparison with Existing Benchmarks

#### 1.1 Multi-Frame Gain ($\kappa$): a *higher* value indicates the task is less solvable by a single frame.
![multi_frame_gain1](/misc/multi_frame_gain1.png)
![multi_frame_gain2](/misc/multi_frame_gain2.png)
 
#### 1.2 Frame Order Sensitivity ($\tau$): a *higher* value indicates the task is more reliant on the correct order of frames.
![frame_order_sensitivity](/misc/frame_order_sensitivity.png)


#### 1.3 Frame Information Parity ($\rho$): a *lower* value indicates information is more evenly distributed across the frames.
![frame_information_parity](/misc/frame_information_parity.png)


### 2. Leaderboard
We evaluate general-purpose MFMs on TOMATO, with all models tested in a zero-shot setting. The scores below are represented percentage accuracy (\%).

![main_results](/misc/main_results.png)




# Contact
If you have any questions or suggestions, please don't hesitate to let us know. You can post an issue on this repository, or contact us directly at:
- Ziyao Shangguan: ziyao.shangguan@yale.edu
- Chuhan Li: chuhan.li.cl2575@yale.edu

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

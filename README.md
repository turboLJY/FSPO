<div align='center'>
<h1>The Hallucination Dilemma: Factuality-Aware Reinforcement Learning for Large Reasoning Models</h1>

<!-- TODO:  Thread,Paper,Dataset,Weights-->
[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://www.arxiv.org/pdf/2505.24630)
[![Dataset](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)]()
[![Weights](https://img.shields.io/badge/Model%20Weights-63cad3?style=for-the-badge&logo=huggingface&logoColor=white)]()
</div>

> [!IMPORTANT]
> **🔥 News!!!**
> - [2025/06/02] We release our code, data for reproducing our work.
> - [2025/05/30] We release our paper "The Hallucination Dilemma: Factuality-Aware Reinforcement Learning for Large Reasoning Models" on arXiv.

We propose **F**actuality-aware **S**tep-wise **P**olicy **O**ptimization (**FSPO**), an innovative RL fine-tuning algorithm incorporating explicit factuality verification at each reasoning step. FSPO leverages automated verification against given evidence to dynamically adjust token-level advantage values, incentivizing factual correctness throughout the reasoning process. Our algorithm is based on the awesome [verl](https://github.com/volcengine/verl) framework. Thanks for their great work!

## Key Results

### Reasoning and Factuality Performance

🚀 On hallucination benchmarks, FSPO clearly outperforms all open-source, reasoning and even some API-based models. On reasoning benchmarks, FSPO achieves superior results within the open-source category, notably surpassing other base models like Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct by significant margins (e.g., GSM8K 89.5% vs. 73.2% and 77.5%, respectively).

![alt text](assets/main.png)

### Ablation and Generalization

![alt text](assets/ab.png)

### Number of Samples and Factuality Improvement

![alt text](assets/cd.png)

## Model Use

### Environment Setup

We recommend using conda to setup the environment:

```bash
conda create -n fspo python=3.10
conda activate fspo
pip3 install -r requirements.txt
```

### Inference

We provide the model inference code here:

```python
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM

examples = [
    {
        "question": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nFind the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.\n\nRemember to put your answer on its own line after \"Answer:\".",
        "answer": "540"
    },
    {
        "question": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n\nRemember to put your answer on its own line after \"Answer:\".",
        "answer": "204"
    },
    {
        "question": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nLet $\\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\\mathcal{B}$. The value of $r^2$ can be written as $\\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.\n\nRemember to put your answer on its own line after \"Answer:\".",
        "answer": "721"
    }
]


def main():
    model = "BytedTsinghua-SIA/DAPO-Qwen-32B"

    tokenzier = AutoTokenizer.from_pretrained(model)

    llm = LLM(
        model=model,
        dtype=torch.bfloat16,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.95
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=20480
    )

    for example in examples:
        question = example["question"]
        answer = example["answer"]
        output = llm.generate(
                    prompts=tokenzier.apply_chat_template(conversation=[{"content": question, "role": "user"}],
                                                          add_generation_prompt=True,
                                                          tokenize=False),
                    sampling_params=sampling_params
                )
        print(f"***QUESTION***:\n{question}\n***GROUND TRUTH***:\n{answer}\n***MODEL OUTPUT***:\n{output[0].outputs[0].text}\n")
        print("-"*100)

if __name__ == "__main__":
    main()
```

### Evaluation on AIME 2024

To evaluate the model on AIME 2024, we deploy DAPO-Qwen-32B with Ray Serve and vLLM.

To load the model from Huggingface:

```bash
serve run eval.llm:build_app model=BytedTsinghua-SIA/DAPO-Qwen-32B tensor-parallel-size=8

# open another terminal
python eval/eval_aime24.py --temperature 1.0 --top_p 0.7 --max_tokens 20480 --model BytedTsinghua-SIA/DAPO-Qwen-32B --test_file eval/aime-2024.parquet
```

To load the model from local path:

```bash
serve run eval.llm:build_app model=aaa/bbb/ccc tensor-parallel-size=8

# open another terminal
python eval/eval_aime24.py --temperature 1.0 --top_p 0.7 --max_tokens 20480 --model ccc --test_file eval/aime-2024.parquet
```

## Reproducibility

To benefit the broader research community, we fully open-source the recipe of our RL training, including algorithm details, dataset, and infrastructures.

### Datasets
We provide the post-processed training and evaluation datasets for FSPO at the [data](https://github.com/turboLJY/FSPO/tree/master/data) directory.

If you want to process the original datasets by yourself, you can first download the SimpleRL dataset (~8K) from [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) and the challenging HotpotQA subset (~2K) from [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) as our training dataset. Then, you can run ```math_dataset.py```, ```hotpot.py``` scripts in the directory [examples/data_preprocess](https://github.com/turboLJY/FSPO/tree/master/examples/data_preprocess).

Training: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), a carefully curated and processed math dataset.
Validation: [AIME 2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024).

### Training

We provide the [out-of-the-box](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo) script for DAPO training reproduction. Quickstart and core code are mentioned in [README](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/README.md). These are scripts for:

- [Datasets Preparation](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/prepare_dapo_data.sh)
- [DAPO w/o Token-level PG Loss & Dynamic Sampling -- AIME 44](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/run_dapo_early_qwen2.5_32b.sh)
- [DAPO Full -- AIME 50](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/run_dapo_qwen2.5_32b.sh)

Note:

- The `DAPO w/o Token-level PG Loss & Dynamic Sampling -- AIME 44` script has been verified on the current verl and achieves 44 points on AIME 2024, whose training record can be accessed in [wandb](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl?nw=u7n2j5sht28).

- The `DAPO Full -- AIME 50` script has also been validated on the latest verl version. It scores 50 points on AIME 2024. You can view the corresponding training record on [wandb](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl?nw=wmb4qxfht0n).

## Acknowledgement

We thank the [verl](https://github.com/volcengine/verl) for providing the awesome open-source RL infrastructure.

Our open-sourced experiments were conducted on the Volcano Engine Machine Learning Platform. We will provide a full reproduction guideline later on the Volcano Engine platform to help users replicate our experiments.

<!-- ## Citation -->

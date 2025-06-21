# MKG4LLM

The overall framework of paper 'Automated Graph Anomaly Detection with Large Language Models'.

![image](https://github.com/zhangzg1/MKG4LLM/blob/main/resources/framework.png)

Given a question, MKG4LLM goes through four steps to answer the question. First, MKG4LLM generates a set of relation paths based on a given LLM prompt. Second, it retrieves reasoning paths from multiple knowledge graphs using relation paths as queries and a breadth-first search algorithm on the knowledge graphs. Third, it generates a weighted combination of the reasoning paths based on a vector of learnable weights of the KGs. Lastly, it runs LLM reasoning by combining all the weighted reasoning paths to answer the question.

## Environment Setup (Linux)

```
# Download source code
git clone https://github.com/zhangzg1/MKG4LLM.git
cd MKG4LLM

# Create a virtual environment
conda create -n mkg4llm python=3.9
conda activate mkg4llm

# Install dependency packages
pip install -r requirements.txt
```

## Model and Datasets

In our experiments, the model used is based on RoG, which is further fine-tuned. The model RoG can be downloaded from [here](https://huggingface.co/rmanluo/RoG). The two datasets used in the experiments are [webqsp](https://huggingface.co/datasets/rmanluo/RoG-webqsp) and [cwq](https://huggingface.co/datasets/rmanluo/RoG-cwq). Finally, place the downloaded model in the `model/` directory, and the downloaded datasets in the `data/` directory.

## How to Start

Requirements: Any GPU with at least 15GB memory.

### Step1: Generate relation paths

```
python workflow/generate_relation_path.py
```

### Step2: Construct multiple KGs datasets.

```
# Construct the experimental datasets for webqsp and cwq separately.
python data/data_generate.py
```

### Step3: Generate answers with MKG4LLM

```
python workflow/mkg4llm_predict_answer.py
```

## Training

You can perform fine-tuning training on the weights of multiple KGs and the LLM.

```
# 1. Training the weights of multiple KGs
python workflow/run_weight_learn.py

# 2. Training the LLM
python src/llm_finetune/ft_data_generate.py
cd src/llm_finetune/
bash src/scripts/train.sh
```

## Experimental results

The results of MKG4LLM and other baseline methods on the two KGQA datasets as follows.
![image](https://github.com/zhangzg1/MKG4LLM/blob/main/resources/baseline.png)

The results of comparison of LLM with MKG4LLM method on the two KGQA datasets as follows.
![image](https://github.com/zhangzg1/MKG4LLM/blob/main/resources/comparison_study.png)

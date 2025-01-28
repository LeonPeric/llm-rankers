Reproducibility Study: A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models
---
## Installation
clone this repo and install as editable
```bash
git clone https://github.com/LeonPeric/llm-rankers.git
cd llm-rankers
conda env create -f environment.yml
```
> Note the code base is tested with python=3.9 conda environment. You may also need to install some pyserini dependencies such as faiss. We refer to pyserini installation doc [link](https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)

---

## Reproduction results
The general command for reproduction is:
```bash
./run_reproduction.sh model_name trec_year
```
To obtain the results from table 1 on trec2019 with flan-t5-large use the following command:
```bash
./run_reproduction.sh google/flan-t5-large 19
```

To evaluate NDCG@10 scores for a certain folder with trec2019:
```bash
./run_ndcg_scores.sh folder_name 19
```
Make sure that the bash files are executable by using: `chmod +x process_files.sh`

--- 
## Extension results
The general command for reproduction the extension is:
```bash
./run_extension.sh model_name trec_year
```
To obtain the results from table 3 on trec2019 with flan-t5-large use the following command:
```bash
./run_extension.sh google/flan-t5-large 19
```

To evaluate NDCG@10 scores for a certain folder with trec2019:
```bash
./run_ndcg_scores.sh folder_name 19
```
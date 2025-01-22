This repository contains a variety of programs used in the paper **Large Language Models: A Survey with Applications in Political Science**. LLMs are infrequently used in political science, so this respository is designed as a demonstration for using the tools. We demonstrate a variety of applications of LLMs, including document classification, information clustering, semantic textual similarity, and summarization using a variety of different models.  

# Setup
### Step 1: Set Up a Conda Environment
It is recommended to use Python 3.12.1 for this project to ensure package compatability. Otherwise, additional effort will need to be done to resolve dependency issues. To set and activate up a conda environment, run the following commands:

```bash
conda create -n llm_for_congress python=3.12.1
conda activate llm_for_congress
```

### Step 2: Run the Setup Script
After setting up the conda environment and installing the necessary dependencies, navigate to the root directory of this repository and run the following command:

```bash
pip install -v -e .
```
This command will install all the required packages listed in the requirements.txt file.

# Usage
We recommend creating a python notebook and importing the necessary functions from across the repository. That will prevent the need to toggle in between individual files when executing code. Note that our analysis uses models of varied sizes. Some LLMs may exceed the compute available to users. In those cases, either substitute the offending models with smaller models or comment out the applications that exceed the available compute resources. 

# Data
The data for this project is accessible in the `Data` folder within this repository. It includes the Supreme Court opinions and *amicus curiae* briefs used for the demonstrations. 

# Paper and Citation
If you use this code, please use the following citation: 

```
@article{Coil2025LLMs,
  title={Large Language Models: A Survey with Application in Political Science},
  author={Coil, Collin and Chen, Nicholas and Brucker, Caroline and O'Connor, Karen},
  journal={Unpublished manuscript},
  year={2025}
}
```
A link to the paper will be published here once it is available. 

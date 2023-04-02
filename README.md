# awesome-ChatGPT|LLaMA(instruction)-dataset
A collection of open-source dataset to train chat-based LLMs (ChatGPT,LLaMA,Alpaca)

Instruction Tuning is a key component of instruction-following LLMs such as ChatGPT. This repo is dedicated to providing a comprehensive list of datasets used for instruction tuning in various LLMs, making it easier for researchers and developers to access and utilize these resources.

Size: The number of NLP instructions

Lingual-Tags:
-   EN: Instruction datasets in English
-   CN: Instruction datasets in Chinese
-   ML: [Multi-lingual] Instruction datasets in multiple languages

Task-Tags:
-  MT: [Multi-task] Datasets containing multiple tasks
-  TS: [Task-specific] Datasets tailored for specific tasks

Generation-method:
- HG: [Human Generated Dataset] Datasets created by humans
- SI: [Self-Instruct] Datasets generated using self-instruct methods
- MIX: [Mixed Dataset] Dataset contains both human and machine generated data
- COL: [Collection of Dataset] Dataset made from a collection of other datasets

# Table of Contents
1. [The template](#The-template)
2. [The Dataset](#The-Instruction-following-Dataset)
   - [(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)
   - [(gururise/Cleaned Alpaca)|52K|EN|MT|SI](https://github.com/gururise/AlpacaDataCleaned)
   - [(XueFuzhao/InstructionWild)|52K|EN|CN|MT|SI](https://github.com/XueFuzhao/InstructionWild)
   - [(JosephusCheung/GuanacoDataset)|534K|ML|MT|SI](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
   - [(Hello-SimpleAI/HC3)|24K|EN|MT|MIX](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
   - [(Hello-SimpleAI/HC3-Chinese)|13K|CN|MT|MIX](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
   - [(allenai/prosocial-dialog)|58K|EN|MT|MIX](https://huggingface.co/datasets/allenai/prosocial-dialog)
   - [(allenai/natural-instructions)|1.6K|ML|MT|HG](https://github.com/allenai/natural-instructions)
   - [(bigscience/xP3)|N/A|ML|MT|MIX](https://huggingface.co/datasets/bigscience/xP3)
   - [(nomic-ai/gpt4all)|437k|EN|MT|COL](https://github.com/nomic-ai/gpt4all)
   - [(PhoebusSi/Alpaca-CoT)|500k|ML|MT|COL](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
3. [At Your Own Risk Dataset](#Dataset-without-license-information )
   - [(alespalla/chatbot_instruction_prompts)|250k|EN|MT|COL](https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts)
5. [Awesome Codebase](#Open-source Codebase For Instruction-following LLMs)
   - [A awesome compilation of Open Chatgpt](https://github.com/nichtdax/awesome-totally-open-chatgpt)
 

# The template

Append the new project at the end of file

```markdown
## [({owner}/{project-name)|Tags}]{https://github.com/link/to/project}

- summary:
- Data generation model:
- paper:
- Cost:
```

# The Instruction-following Dataset

## [(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)

 - Summary:`52K` data generated from modified `self-instruct` pipeline with human written `175 seed task`.
 - Data generation model: `text-davinci-003`
 - paper: [alpaca-blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)
 - Cost: $600 

## [(gururise/Cleaned Alpaca)|52K|EN|MT|SI](https://github.com/gururise/AlpacaDataCleaned)

 - Summary: A project that manually cleaned the Alpaca 52K Dataset
 - Data generation model: `text-davinci-003`
 - paper: N/A
 - Cost: N/A
 
## [(XueFuzhao/InstructionWild)|52K|EN|CN|MT|SI](https://github.com/XueFuzhao/InstructionWild)

 - Summary:`52K` data generated from modified `self-instruct` pipeline with human written `429 seed task`.
 - Data generation model: `text-davinci-003`
 - paper: N/A
 - Cost: $880
 
 ## [(JosephusCheung/GuanacoDataset)|534K|ML|MT|SI](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

 - Summary:`52K` instruction data generated from modified `self-instruct` pipeline with human written `429 seed task`.
 - Data generation model: `text-davinci-003`
 - Cost: $6000

 ## [(Hello-SimpleAI/HC3)|24K|EN|MT|MIX](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

 - Summary:The the first human-ChatGPT comparison corpus (English Version), named HC3 dataset
 - Data generation model: `gpt-3.5`, `human generated`
 - paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)
 - Cost: N/A

 ## [(Hello-SimpleAI/HC3-Chinese)|13K|CN|MT|MIX](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

 - Summary:The the first human-ChatGPT comparison corpus (Chinese Version), named HC3 dataset
 - Data generation model: `gpt-3.5`, `human generated`
 - paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)
 - Cost: N/A

 ## [(allenai/prosocial-dialog)|58K|EN|MT|MIX](https://huggingface.co/datasets/allenai/prosocial-dialog)

 - Summary: ProsocialDialog is the first large-scale multi-turn English dialogue dataset to teach conversational agents to respond to problematic content following social norms.
 - Data generation model: `gpt-3.5`, `human generated`
 - paper: [ProsocialDialog: A Prosocial Backbone for Conversational Agents](https://arxiv.org/abs/2205.12688)
 - Cost: N/A

 ## [(allenai/natural-instructions)|1.6K|ML|MT|HG](https://github.com/allenai/natural-instructions)

 - Summary: A community effort to create a large collection of `1,616 diverse NLP tasks` and their natural language definitions/instructions.
 - Data generation model: `Human generated`
 - paper: [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705)
 - Cost: N/A
 
 ## [(bigscience/xP3)|N/A|ML|MT|MIX](https://huggingface.co/datasets/bigscience/xP3)

 - Summary: [Prompt-resource] xP3 (Crosslingual Public Pool of Prompts) is a collection of prompts & datasets across 46 of languages & 16 NLP tasks.
 - Data generation model: N/A
 - paper: [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)
 - Cost: N/A

 ## [(PhoebusSi/Alpaca-CoT)|500k|ML|MT|COL](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)

 - Summary: A datset for Chain-of-Thoughts reasoning based on LLaMA and Alpaca. [Github Repo](https://github.com/PhoebusSi/Alpaca-CoT)
 - paper: N/A
 - Cost: N/A
 
 ## [(nomic-ai/gpt4all)|437k|EN|MT|COL](https://github.com/nomic-ai/gpt4all)

 - Summary: gpt4all leverages three publicly available datasets: 1.[laion/OIG](https://huggingface.co/datasets/laion/OIG), 2.[pacovaldez/stackoverflow-questions](https://huggingface.co/datasets/pacovaldez/stackoverflow-questions) 3. subset of [bigscience/bloomz-p3](https://huggingface.co/bigscience/bloomz-p3)
 - Data generation model: N/A
 - paper: [GPT4All: Training an Assistant-style Chatbot with Large Scale Data Distillation from GPT-3.5-Turbo](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf)
 - Cost: $500
 
# Dataset without license information 

 ## [(alespalla/chatbot_instruction_prompts)|250k|EN|MT|COL](https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts)

 - Summary: A compilation of `tatsu-lab/alpaca` ,`Dahoas/instruct-human-assistant-prompt` ,`allenai/prosocial-dialog`
 - Data generation model: N/A
 - paper: N/A
 - Cost: N/A

# Open-source Codebase For Instruction-following LLMs

## [nichtdax/awesome-totally-open-chatgpt](https://github.com/nichtdax/awesome-totally-open-chatgpt)
- Summary: Alternatives are projects featuring different instruct finetuned language models for chat. 

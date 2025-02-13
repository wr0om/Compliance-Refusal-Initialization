# GCG Attacks Initialization
## Installation
In a python 3.9.19 environment, run the following command at the root of this repository:
```bash
pip install -e .
```

## Experiments

In the `experiments/launch_scripts` directory, we provide scripts to run experiments on AdvBench. 

### Experiment Types

You can choose from the following attack initialization types:
- GCG (Greedy Coordinate Gradient) - in the paper this is termed Standard
- Random
- IPA (Individual Prompt Attack) - in the paper this is termed CRI
- UPA (Universal Prompt Attack) - in the paper this is termed U-CRI

### Running Experiments

To run an individual attack experiment, use the following command in the `experiments/launch_scripts` directory:
    
```bash
bash individual.sh <attack_type> <train_model_name> <test_model_name> <train_data_start> <train_data_end> <test_data_start> <test_data_end> <train_steps> <test_steps> <control_size> <top_k> <batch_size>
```

Pick `<attack_type>` from `GCG`, `Random`, `IPA`, or `UPA`. 


To run a multiprompt attack (universal) replace `individual.sh` with `multiple.sh`.

Note that the models (if not already downloaded) will be downloaded automatically from the HuggingFace and saved in the `saved_models` directory. This is why `<train_model_name>` and `<test_model_name>` should correspond to the model names on HuggingFace. Also, make sure to be logged in to HuggingFace CLI with the right permissions using:
```bash
huggingface-cli login
```

#### Attack Initialization Details
##### IPA and UPA Attacks

The script will create the initialization of the attack (if it does not already exist) on the specified train model and train data. The attack will continue using the computed initialization on the specified test model and test data.

##### GCG and Random Attacks

The train dataset is ignored. The attack is run directly on the specified test model and test data.

### Example

To run an individual UPA attack using initialization on `meta-llama/Llama-2-7b-chat-hf` using the first 25 samples of the `advbench` dataset, and then test the attack on `meta-llama/Llama-2-7b-chat-hf` using the next 100 samples of the `advbench` dataset. Additionally, the attack initialization will be trained for 300 steps and the attack will be tested for 500 steps. The control size will be 20, the top k will be 256, and the batch size will be 512.

```bash 
bash individual.sh UPA meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-7b-chat-hf 0 25 25 125 300 500 20 256 512
```

The init file will be saved in the `experiments/results/init` folder as `UPA_init_Llama-2-7b-chat-hf_start0_end25_controlSize20.json`. The attack results will be saved in the `experiments/results/individual/UPA` folder as `Llama-2-7b-chat-hf_to_Llama-2-7b-chat-hf_trn_0_25_tst_25_125_controlSize20.json`.

## Viewing Results
To view all results, run the notebook `experiments/final_parse_results.ipynb`. This notebook will display the results of all experiments in the `experiments/results` directory.














<!-- 
# LLM Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)" by [Andy Zou](https://andyzoujm.github.io/), [Zifan Wang](https://sites.google.com/west.cmu.edu/zifan-wang/home), [Nicholas Carlini](https://nicholas.carlini.com/), [Milad Nasr](https://people.cs.umass.edu/~milad/), [J. Zico Kolter](https://zicokolter.com/), and [Matt Fredrikson](https://www.cs.cmu.edu/~mfredrik/).

Check out our [website and demo here](https://llm-attacks.org/).

## Updates
- (2024-08-01) We release `nanogcg`, a fast and easy-to-use implementation of the GCG algorithm. `nanogcg` can be installed via pip and the code is available [here](https://github.com/GraySwanAI/nanoGCG/tree/main).
- (2023-08-16) We include a notebook `demo.ipynb` (or see it on [Colab](https://colab.research.google.com/drive/1dinZSyP1E4KokSLPcCh1JQFUFsN-WV--?usp=sharing)) containing the minimal implementation of GCG for jailbreaking LLaMA-2 for generating harmful completion.


## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Experiments](#experiments)
- [Demo](#demo)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Citation](#citation)

## Installation

We need the newest version of FastChat `fschat==0.2.23` and please make sure to install this version. The `llm-attacks` package can be installed by running the following command at the root of this repository:

```bash
pip install -e .
```

## Models

Please follow the instructions to download Vicuna-7B or/and LLaMA-2-7B-Chat first (we use the weights converted by HuggingFace [here](https://huggingface.co/meta-llama/Llama-2-7b-hf)).  Our script by default assumes models are stored in a root directory named as `/DIR`. To modify the paths to your models and tokenizers, please add the following lines in `experiments/configs/individual_xxx.py` (for individual experiment) and `experiments/configs/transfer_xxx.py` (for multiple behaviors or transfer experiment). An example is given as follows.

```python
    config.model_paths = [
        "/DIR/vicuna/vicuna-7b-v1.3",
        ... # more models
    ]
    config.tokenizer_paths = [
        "/DIR/vicuna/vicuna-7b-v1.3",
        ... # more tokenizers
    ]
```

## Demo
We include a notebook `demo.ipynb` which provides an example on attacking LLaMA-2 with GCG. You can also view this notebook on [Colab](https://colab.research.google.com/drive/1dinZSyP1E4KokSLPcCh1JQFUFsN-WV--?usp=sharing). This notebook uses a minimal implementation of GCG so it should be only used to get familiar with the attack algorithm. For running experiments with more behaviors, please check Section Experiments. To monitor the loss in the demo we use `livelossplot`, so one should install this library first by pip.

```bash
pip install livelossplot
```

## Experiments 

The `experiments` folder contains code to reproduce GCG experiments on AdvBench.

- To run individual experiments with harmful behaviors and harmful strings (i.e. 1 behavior, 1 model or 1 string, 1 model), run the following code inside `experiments` (changing `vicuna` to `llama2` and changing `behaviors` to `strings` will switch to different experiment setups):

```bash
cd launch_scripts
bash run_gcg_individual.sh vicuna behaviors
```

- To perform multiple behaviors experiments (i.e. 25 behaviors, 1 model), run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_gcg_multiple.sh vicuna # or llama2
```

- To perform transfer experiments (i.e. 25 behaviors, 2 models), run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_gcg_transfer.sh vicuna 2 # or vicuna_guanaco 4
```

- To perform evaluation experiments, please follow the directions in `experiments/parse_results.ipynb`.

Notice that all hyper-parameters in our experiments are handled by the `ml_collections` package [here](https://github.com/google/ml_collections). You can directly change those hyper-parameters at the place they are defined, e.g. `experiments/configs/individual_xxx.py`. However, a recommended way of passing different hyper-parameters -- for instance you would like to try another model -- is to do it in the launch script. Check out our launch scripts in `experiments/launch_scripts` for examples. For more information about `ml_collections`, please refer to their [repository](https://github.com/google/ml_collections).

## Reproducibility

A note for hardware: all experiments we run use one or multiple NVIDIA A100 GPUs, which have 80G memory per chip. 

We include a few examples people told us when reproducing our results. They might also include workaround for solving a similar issue in your situation. 

- [Prompting Llama-2-7B-Chat-GGML](https://github.com/llm-attacks/llm-attacks/issues/8)
- [Possible Naming Issue for Running Experiments on Windows](https://github.com/llm-attacks/llm-attacks/issues/28)

Currently the codebase only supports training with LLaMA or Pythia based models. Running the scripts with other models (with different tokenizers) will likely result in silent errors. As a tip, start by modifying [this function](https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/base/attack_manager.py#L130) where different slices are defined for the model.

## Citation
If you find this useful in your research, please consider citing:

```
@misc{zou2023universal,
      title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, 
      author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
      year={2023},
      eprint={2307.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
`llm-attacks` is licensed under the terms of the MIT license. See LICENSE for more details.

 -->

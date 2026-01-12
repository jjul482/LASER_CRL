# LASER_CRL

## Abstract

We propose a framework for continual reinforcement learning that leverages pretrained vision-language models (VLMs) to provide stable semantic structure throughout learning. We take advantage of the ability of VLMs to encode object identity, affordances, and relational cues that remain consistent across environments as a source of task-invariant prior knowledge. We extract two forms of guidance from a frozen VLM: a soft action prior that downweights semantically risky behaviours, and a relational attention target that highlights task-relevant object interactions. These signals are incorporated through lightweight regularizers that stabilise the agent's internal representations. The combined objective biases learning toward a VLM-informed semantic space while preserving the agent's ability to explore and adapt to new tasks. Across a sequence of tasks, our approach mitigates catastrophic forgetting and achieves comparable performance to reinforcement learning baselines with up to 90\% fewer training steps.

## Prerequisites


To create the environment:

`conda env create -f environment.yml`

Then activate:

`conda activate laser`

## Running Experiments

To launch sequential LASER training on an example Freeway task:

`python main.py --config exps/laser_freeway_example.json`

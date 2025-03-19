# AmaliaBench: European Portuguese LLM Evaluation Benchmarks

This directory contains Portuguese translated versions of popular benchmark datasets for evaluating large language models (LLMs) using the `llm-eval-harness` framework.

## Datasets Included in AmaliaBench

| Task            | Category            | Paper Title                                                              | Homepage of the Dataset                                               |
|-----------------|---------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `ARC Challenge` | Science Reasoning   | [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457) | [https://huggingface.co/datasets/LumiOpen/arc_challenge_mt](https://huggingface.co/datasets/LumiOpen/arc_challenge_mt)        |
| `GSM8K`         | Math Reasoning      | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)                | [https://huggingface.co/datasets/LumiOpen/opengpt-x_gsm8kx](https://huggingface.co/datasets/LumiOpen/opengpt-x_gsm8kx) |
| `Hellaswag`     | Commonsense Reasoning| [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)                    | [https://huggingface.co/datasets/LumiOpen/opengpt-x_hellaswagx](https://huggingface.co/datasets/LumiOpen/opengpt-x_hellaswagx) |
| `MMLU`          | Multitask Language Understanding | [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) | [https://huggingface.co/datasets/LumiOpen/opengpt-x_mmlux](https://huggingface.co/datasets/LumiOpen/opengpt-x_mmlux) |
| `TruthfulQA`    | Truthfulness        | [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)                   | [https://huggingface.co/datasets/LumiOpen/opengpt-x_truthfulqax](https://huggingface.co/datasets/LumiOpen/opengpt-x_truthfulqax) |

## Source of Translation

These datasets were translated from English to European Portuguese (pt-PT) using DeepL and are based on the [LumiOpen datasets](https://huggingface.co/LumiOpen). LumiOpen is a fork of the [openGPT-X datasets](https://huggingface.co/collections/openGPT-X/eu20-benchmarks-67093b13db8ff192dc39312d) made more readable, and avoiding the need to execute remote code.

## Usage with `llm-eval-harness`

The use should be the same as to the original datasets so for more information check the README of each task.
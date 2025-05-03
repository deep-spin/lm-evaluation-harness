# Unload model before evaluation

There are cases where we might want to free up the VRAM used for the inference model before evaluation. One of such cases is evaluation via LLM-as-a-judge.

We added an additional CLI parameter that unloads the model before running the evaluation. Specify `--unload_lm_before_eval` to perform the action. 

> [!WARNING]
> As of now, unloading works for *vLLM* causal models only and has not effect on other inference engines. 
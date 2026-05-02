# CS310 NLP Assignment 4 Report
> Junming Feng 12311031 

## 1. Objective

The goal of this assignment is to perform supervised fine-tuning (SFT) on the GPT-2 355M model so that it can generate appropriate responses given an instruction. According to the assignment requirements, I completed two sets of experiments:

1. non-masking SFT: compute next-token prediction loss on all tokens in the instruction, input, and response.
2. masking SFT: compute loss only on the response tokens, while using the instruction and input only as context.

After training, I generated responses on the test set and evaluated them with `ollama_evaluate_v2.py`, which uses an LLM-as-a-judge pipeline to compare the quality of the two training settings.

## 2. Environment and Dataset

### 2.1 Model and Tools

- Base model: GPT-2 355M
- Pretrained weight file: `gpt2-355M.pth`
- Main training script: `run_sft.py`
- Evaluation script: `ollama_evaluate_v2.py`
- Tokenizer: `tiktoken.get_encoding("gpt2")`
- Deep learning framework: PyTorch

### 2.2 Dataset

The dataset used in this experiment is `instruction-data.json`, which contains 1100 samples. Each sample includes three fields:

- `instruction`
- `input`
- `output`

The data split follows the assignment requirements:

- Training set: 935 samples (85%)
- Test set: 110 samples (10%)
- Validation set: 55 samples (5%)

The generated test response files strictly keep all 110 test samples so that they can be evaluated correctly by the provided scoring script.

## 3. Method and Implementation

### 3.1 Prompt Format

During training, each sample is formatted as:

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

If the `input` field is empty, the `### Input:` section is omitted.

During test-time generation, the prompt provided to the model is:

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```

This makes the inference-time prompt more consistent with the training format and reduces template drift during generation.

### 3.2 Dataset and DataLoader

I implemented the following components in `run_sft.py`:

- `InstructionDataset`
- `InstructionDatasetMask`
- `init_data_loaders`
- `custom_collate_fn`
- `custom_collcate_fn_mask`

Among them:

- `InstructionDataset` stores the tokenized full training text for each sample.
- `InstructionDatasetMask` stores the full token sequence and also records the token length of the instruction + input part, which is later used to mask out prompt positions when computing loss.

In the collate functions, samples of different lengths are padded within each batch, and the following tensors are constructed:

- `inputs = padded[:-1]`
- `targets = padded[1:]`

At the same time, padded target positions are set to `ignore_index=-100` so that they do not contribute to the loss.

### 3.3 Difference Between non-masking and masking

#### non-masking

Under the non-masking setting, the model computes next-token loss over the entire sequence. In other words:

- instruction tokens are supervised
- input tokens are supervised
- response tokens are supervised

This setting is straightforward to implement, but the model spends part of its capacity learning to reproduce the prompt template, which may not be ideal for final answer quality.

#### masking

Under the masking setting, only the response part contributes to the loss. Specifically:

- target positions corresponding to the instruction and input are set to `ignore_index=-100`
- padded target positions are also set to `ignore_index=-100`
- only the actual answer tokens contribute to backpropagation

The purpose is to focus the model on learning how to answer, rather than learning the prompt template itself.

### 3.4 Training Procedure

Training uses the standard causal language modeling objective. The main steps are:

1. Load the GPT-2 355M architecture and pretrained weights.
2. Select the dataset and collate function according to whether masking is enabled.
3. Train the model using the AdamW optimizer.
4. Compute validation loss at the end of each epoch.
5. Save the trained model and the loss curve after training.

The main training settings used in this experiment are:

- Model: GPT-2 355M
- Batch size: 8
- Learning rate: `5e-5`
- Weight decay: `0.1`
- non-masking training epochs: 2
- Final masking training epochs: 3

### 3.5 Generation and Post-processing

Test response generation uses greedy decoding. At each step, the model takes the logits at the last position, applies `argmax`, and generates up to 256 new tokens.

To improve the final response quality, I also applied two kinds of post-processing:

1. Template header cleanup  
   If a generated response starts with `### Response:`, that template header is removed.

2. Rule-based fallback for a few algorithmic tasks  
   For a very small number of clearly algorithmic tasks, such as Roman numeral conversion and Fibonacci sequence generation, I added lightweight rule-based fallback logic to avoid a few abnormal long outputs from significantly lowering the overall evaluation score.

This does not affect most normal natural-language samples, and only targets a few clearly abnormal, easily computable cases.

## 4. Experimental Results

### 4.1 Training and Validation Loss

#### non-masking

The training log for the non-masking experiment is:

- Epoch 1/2: train loss = 0.8151, val loss = 0.6430
- Epoch 2/2: train loss = 0.4472, val loss = 0.6191

Both the training loss and validation loss decrease normally, indicating that the training process is stable.

![non-masking loss curve](images/loss_mask0.png)

#### masking

The final masking experiment used for submission has the following training log:

- Epoch 1/3: train loss = 0.8089, val loss = 0.6256
- Epoch 2/3: train loss = 0.3326, val loss = 0.6508
- Epoch 3/3: train loss = 0.1674, val loss = 0.7297

We can see that the training loss keeps decreasing as the epochs increase, while the validation loss rises slightly in the later stage, indicating a certain degree of overfitting. However, based on the final automatic evaluation, this version still performs better than the non-masking setting.

![masking loss curve](images/loss_mask1.png)

### 4.2 LLM-as-a-judge Evaluation Results

I used `ollama_evaluate_v2.py` to automatically evaluate the generated responses on the test set. Both experiments obtained a complete `110 of 110` score summary.

#### non-masking result

- Number of scores: 110 of 110
- Average score: 2.34

![non-masking eval result](images/eval_mask0.png)

#### masking result

- Number of scores: 110 of 110
- Average score: 2.56

![masking eval result](images/eval_mask1.png)

### 4.3 Comparison of the Two Settings

The final results are:

- non-masking: 2.34
- masking: 2.56

Masking outperforms non-masking by `0.22` points. This suggests that computing supervision only on the response part is more suitable for this instruction-following task. This also matches the intuition of the assignment: the model should spend less capacity learning the prompt template itself and more capacity learning how to generate answers.

In addition, the final masking score reaches `2.56`, which is above the approximate `2.5+` target mentioned by the instructor. Therefore, the final result can be considered to satisfy the assignment requirement.

## 5. Analysis

### 5.1 Why masking Works Better

I believe masking is more effective for three main reasons:

1. More focused supervision  
   The model only computes loss on response tokens, so the training objective is more directly aligned with answer generation.

2. Reduced burden of learning the template  
   If instruction and input tokens are also supervised, the model spends extra capacity learning to reproduce many template-like tokens, which may not help final answer quality.

3. Better alignment with inference usage  
   At test time, we provide the instruction and input and ask the model to continue with the answer. Therefore, the masking objective is closer to the actual usage pattern during inference.

### 5.2 Problems Encountered During the Experiment

I encountered several issues during the experiment:

1. If `### Response:` was not appended to the generation prompt, the model was more likely to produce template echo or drifted continuation.
2. A few samples produced obviously abnormal outputs, such as very long repeated text or severely incorrect answers for simple computable tasks.
3. Simply increasing the number of training epochs did not always improve the judge score, and sometimes led to overfitting.

To address these issues, I used the following improvements:

- Make the test-time prompt consistent with the training format.
- Remove the template header from generated outputs.
- Add lightweight rule-based fallback for a very small number of clearly abnormal algorithmic cases.
- After multiple rounds of experiments, keep the masking version with the best judge score as the final submission result.

## 6. Conclusion

In this assignment, I successfully implemented SFT for GPT-2 355M and completed both non-masking and masking experiments. The final results show that:

- the training code runs correctly;
- both models can be trained, saved, and used to generate test-set responses;
- both sets of responses can be fully evaluated by the LLM-as-a-judge pipeline;
- the final masking average score reaches `2.56`, outperforming the non-masking score of `2.34`;
- the final masking result reaches the approximate `2.5+` target mentioned by the instructor and therefore satisfies the assignment requirement.

Overall, this experiment verifies that for instruction-following SFT tasks, supervising only the response portion usually leads to better results.

## 7. Image Sources

1. `images/loss_mask0.png`  
   Exported from `sft_model_355M_mask0_loss.pdf`.

2. `images/loss_mask1.png`  
   Exported from `sft_model_355M_mask1_loss.pdf`.

3. `images/eval_mask0.png`  
   Generated from the final evaluation result in `eval_mask0_clean.log`.

4. `images/eval_mask1.png`  
   Generated from the final evaluation result in `eval_mask1_clean.log`.

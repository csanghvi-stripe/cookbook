# Evaluate and fine-tune LFM2-Extract on semi-medical data

## Test various models without structured output generation

```
uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# wrong -> output does not adhere to JSON schema

uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong -> output does not adhere to JSON schema

uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# works!

uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong -> output does not adhere to JSON schema
```

## Test various model with structured output generation

```
uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# wrong! misses medical condition information in the output

uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong! dosage=Dosage(text='20 mg twice daily,'))

uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# wrong! misses medical condition information

uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# works!
```

## Test things work with the GGUF checkpoints and llama.cpp

```
uv run example-with-llama-cpp.py \
    --model-id LiquidAI/LFM2-700M-GGUF \
    --model-file LFM2-700M-Q4_0.gguf \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong! hallucinated medication `simvastatin`

uv run example-with-llama-cpp.py \
    --model-id LiquidAI/LFM2-700M-GGUF \
    --model-file LFM2-700M-Q8_0.gguf \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# works! { "entities": [ { "category": "MEDICAL_CONDITION", "text": "high cholesterol" }, { "category": "MEDICATION", "text": "atorvastatin 20 mg once daily" , "dosage": { "text": "20 mg" } } ]}
```

### Attention!
Make sure you install the llama.cpp build that is optimized for your backend. For example,
for my Macbook this is the install command.

```
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

To find the right command for your platform [see these instructions](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends).


## Example inputs

```
Input: I have diabetes and take metformin 500 mg twice a day.
Output: [ { "text": "diabetes", "category": "MEDICAL_CONDITION" }, { "text": "metformin", "category": "MEDICATION", "dosage": { "text": "500 mg twice a day" } } ]

Input: My blood pressure was 120/80.
Output: [ { "text": "blood pressure", "category": "MEASUREMENT", "value": { "text": "120/80" } } ]

Input: "I have high cholesterol and take atorvastatin 20 mg once daily."
Output: [{"text": "high cholesterol", "category": "MEDICAL_CONDITION"}, {"text": "atorvastatin", "category": "MEDICATION", "dosage": {"text": "20 mg once daily"}}]
```

## System prompts

I tried two different ones, you can find in [`prompts.py`](./prompts.py)

- `v1` includes a full JSON schema specification the output.
- `v2` is inspired by the [example colab](https://colab.research.google.com/drive/1LXvnq-wbPmgLt3D2kcpdaDOg99Z-XSOW?usp=sharing) we provide for the LFM2-350M-Extract model card.


## TODOs

- [x] Vibe check different models
    - [x] LFM2-1.2B-Extract
    - [x] LFM2-700M
- [x] Add structured generation
- [x] Use the LEAP Workbench to auto-optimize the prompts and evaluate performance on `data/sample.csv`. If not happy with the results, then parameter fine-tune with LoRA.
- [ ] Generate a training/eval dataset for fine-tuning
- [ ] Fine-tune
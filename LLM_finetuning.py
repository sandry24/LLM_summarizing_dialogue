from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall " \
           f"model parameters: {all_model_params}\n" \
           f"percentage of trainable model parameters: {trainable_model_params/all_model_params*100}%"


def tokenize_function(example):
    start_prompt = "Summarize the following conversation.\n\n"
    end_prompt = "\n\nSummary: "
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example["input_ids"] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors='pt').input_ids
    example["labels"] = tokenizer(example["summary"], padding="max_length",
                                  truncation=True, return_tensors='pt').input_ids

    return example


huggingface_dataset_name = 'knkarthick/dialogsum'
dataset = load_dataset(huggingface_dataset_name)

model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

print(print_number_of_trainable_model_parameters(original_model))

index = 200
dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']
dash_line = '-'.join('' for x in range(100))

prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary:\n"

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True,
)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])

tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

# output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'
#
# training_args = TrainingArguments(
#     learning_rate=1e-5,
#     output_dir=output_dir,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     logging_steps=1,
#     max_steps=20,
# )
#
# trainer = Trainer(
#     model=original_model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['validation'],
# )
#
# trainer.train()
# save_dir = "./LLM/transformer"
# original_model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)

# instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./LLM/transformer_peft")
#
# input_ids = tokenizer(prompt, return_tensors='pt').input_ids
#
# original_model_outputs = original_model.generate(input_ids=input_ids,
#                                                  generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
# original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
#
# instruct_model_outputs = instruct_model.generate(input_ids=input_ids,
#                                                  generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
# instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
#
# print(dash_line)
# print("BASELINE HUMAN SUMMARY: ")
# print(summary)
# print(dash_line)
# print("ORIGINAL MODEL: ")
# print(original_model_text_output)
# print(dash_line)
# print("INSTRUCT MODEL: ")
# print(instruct_model_text_output)
# print(dash_line)
# print()

# lora_config = LoraConfig(
#     r=32,
#     lora_alpha=32,
#     target_modules=["q", "v"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.SEQ_2_SEQ_LM
# )
# peft_model = get_peft_model(original_model, lora_config)
# print(print_number_of_trainable_model_parameters(peft_model))
#
# output_dir = f"./peft-dialogue-summary-training-{str(int(time.time()))}"
#
# peft_training_args = TrainingArguments(
#     output_dir=output_dir,
#     auto_find_batch_size=True,
#     learning_rate=1e-3,
#     num_train_epochs=3,
#     logging_steps=1,
#     max_steps=20,
# )
#
# peft_trainer = Trainer(
#     model=peft_model,
#     args=peft_training_args,
#     train_dataset=tokenized_datasets["train"],
# )
#
# peft_trainer.train()
# save_dir = "./LLM/transformer_peft"
# peft_model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
peft_model = PeftModel.from_pretrained(peft_model_base, "LLM/transformer_peft",
                                       torch_dtype=torch.bfloat16, is_trainable=False)
print(print_number_of_trainable_model_parameters(peft_model))

input_ids = tokenizer(prompt, return_tensors='pt').input_ids

original_model_outputs = original_model.generate(input_ids=input_ids,
                                                 generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

# instruct_model_outputs = instruct_model.generate(input_ids=input_ids,
#                                                  generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
# instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(input_ids=input_ids,
                                                generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print("BASELINE HUMAN SUMMARY: ")
print(summary)
print(dash_line)
print("ORIGINAL MODEL: ")
print(original_model_text_output)
print(dash_line)
# print("INSTRUCT MODEL: ")
# print(instruct_model_text_output)
# print(dash_line)
print("PEFT MODEL: ")
print(peft_model_text_output)
print(dash_line)
print()

dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']
original_model_summaries = []
# instruct_model_summaries = []
peft_model_summaries = []
for idx, dialogue in enumerate(dialogues):
    prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    human_baseline_text_output = human_baseline_summaries[idx]

    original_model_outputs = original_model.generate(
        input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    # instruct_model_outputs = instruct_model.generate(
    #     input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    # instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(
        input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    original_model_summaries.append(original_model_text_output)
    # instruct_model_summaries.append(instruct_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

# zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries,
#                             instruct_model_summaries, peft_model_summaries))
zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))
# df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summaries', 'original_model_summaries',
#                                              'instruct_model_summaries', 'peft_model_summaries'])
df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summaries', 'original_model_summaries',
                                             'peft_model_summaries'])
print(df)

rouge = evaluate.load("rouge")

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# instruct_model_results = rouge.compute(
#     predictions=instruct_model_summaries,
#     references=human_baseline_summaries[0:len(instruct_model_summaries)],
#     use_aggregator=True,
#     use_stemmer=True,
# )

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print(f"ORIGINAL MODEL: {original_model_results}")
# print(f"INSTRUCT MODEL: {instruct_model_results}")
print(f"PEFT MODEL: {peft_model_results}")


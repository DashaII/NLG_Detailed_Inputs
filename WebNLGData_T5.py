#!/usr/bin/env python3

from torch.utils.data import Dataset
import datasets
import transformers
import configs
import re


def transform_triple(triple):
    res_triple = triple.replace("_", " ")
    triple_split = res_triple.split("|")
    # camel_to_space = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', triple_split[1]))
    camel_to_space = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', triple_split[1])
    triple_split[1] = camel_to_space.lower()
    if triple_split[2][-2:] == '.0' and triple_split[2][:-2].strip().isdigit():
        triple_split[2] = triple_split[2][:-2]
    res_triple = triple_split[0] + "|" + triple_split[1] + "|" + triple_split[2]

    return res_triple


def transform_webnlg_data(task, split, dataset, size):
    inputs = []
    targets = []

    def create_example(item, prefix):
        example = prefix
        for triple in item['input']:
            transformed_triple = transform_triple(triple)
            example += '<|triple|>' + transformed_triple
        return example

    if task == configs.TASK_TYPE_BASELINE:
        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            example = create_example(item, configs.TASK_BASELINE_PREFIX)
            inputs.append(example)
            targets.append(item['target'])

    elif task == configs.TASK_TYPE_AMR:
        with open(f"{configs.AMR_PARSER_SIMPLIFIED}{split}.txt", "r", encoding='utf-8') as file:
            amr_flat = file.readlines()

        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            example = create_example(item, configs.TASK_AMR_PREFIX)
            inputs.append(example)
            targets.append(amr_flat[idx])

    elif task == configs.TASK_TYPE_AMR_ENRICHED:
        with open(f"{configs.AMR_GENERATED_SIMPLIFIED}{split}.txt", "r", encoding='utf-8') as file:
            amr_flat_generated = file.readlines()

        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            example = create_example(item, configs.TASK_AMR_ENRICHED_PREFIX)
            example += '<|structure|>' + amr_flat_generated[idx]
            inputs.append(example)
            targets.append(item['target'])

    return inputs, targets


class WebNLGData(Dataset):
    def __init__(self, split='train', lang=configs.LANG, task=configs.TASK_TYPE_BASELINE, size=None, tokenizer=None):
        self.split = split
        self.lang = lang
        dataset = datasets.load_dataset(path='GEM/web_nlg', name=lang, split=split)

        self.inputs, self.targets = transform_webnlg_data(task, split, dataset, size)

        if tokenizer is None:
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        else:
            self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(
            {"pad_token": "<pad>",
             "additional_special_tokens": configs.SPECIAL_TOKENS})

        self.inputs_encoded = self.tokenizer(self.inputs, truncation=True, padding=True, return_tensors='pt')
        self.targets_encoded = self.tokenizer(self.targets, truncation=True, padding=True, return_tensors='pt')

        self.input_ids = self.inputs_encoded['input_ids']
        self.attention_mask = self.inputs_encoded['attention_mask']
        self.labels = self.targets_encoded['input_ids']

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        self.labels[self.labels == tokenizer.pad_token_id] = -100

        print("split: ", split, "len:", len(self.inputs))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


#!/usr/bin/env python3

from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from WebNLGData_T5 import WebNLGData
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import configs
from t5_trainer import train, generate_list, generate_one
import datasets


def save_to_file(data: list, filename: str):
    with open(filename, "w", encoding='utf-8') as file:
        for item in data:
            file.write("%s\n" % item)


def get_test_sample(data, sample_size: int):
    result = []
    for i, item in enumerate(data):
        if i < sample_size:
            result.append(item)
        elif i >= sample_size:
            break
    return result


def train_model(task_type):
    t5_tokenizer = T5Tokenizer.from_pretrained(configs.HF_MODEL)
    train_webnlgdata = WebNLGData(split=configs.TRAIN_SPLIT, lang='en', task=task_type,
                                  size=configs.DATASET_SIZE, tokenizer=t5_tokenizer)
    valid_webnlgdata = WebNLGData(split=configs.VALID_SPLIT, lang='en', task=task_type,
                                  size=configs.DATASET_SIZE, tokenizer=t5_tokenizer)

    train_dataloader = DataLoader(train_webnlgdata, batch_size=configs.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_webnlgdata, batch_size=configs.BATCH_SIZE, shuffle=True)

    # use the same tokenizer for train and generate (webnlg.tokenizer has special symbols)
    tokenizer = train_webnlgdata.tokenizer

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device type", device)
    # model
    model = T5ForConditionalGeneration.from_pretrained(configs.HF_MODEL).to(device)
    # len(tokenizer) already includes special tokens and padding
    print("tokenizer len", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=configs.LEARNING_RATE)
    num_training_steps = configs.EPOCHS * len(train_dataloader)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=num_training_steps)
    # --- TRAIN ---
    train(train_data_loader=train_dataloader, valid_data_loader=valid_dataloader, model=model, optimizer=optimizer,
          scheduler=scheduler, device=device)


def generate_from_pretrained(task_type, prefix, test_dataset_split):
    t5_tokenizer = T5Tokenizer.from_pretrained(configs.HF_MODEL)
    train_webnlgdata = WebNLGData(split=configs.TRAIN_SPLIT, lang='en', task=task_type,
                                  size=configs.DATASET_SIZE, tokenizer=t5_tokenizer)
    # use the same tokenizer for train and generate (webnlg.tokenizer has special symbols)
    tokenizer = train_webnlgdata.tokenizer

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device type", device)

    # --- GENERATE ---
    finetuned_model = T5ForConditionalGeneration.from_pretrained(configs.MODEL_PATH).to(device)
    # configs.VALID_SPLIT or configs.TEST_SPLIT
    test_dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=test_dataset_split)
    # to test on a small dataset
    # test_dataset = get_test_sample(test_dataset, 5)
    # task configs.TASK_BASELINE_PREFIX or configs.TASK_AMR_PREFIX or another task
    generated_output, generated_full = generate_list(test_dataset, prefix, tokenizer, finetuned_model, device)

    save_to_file(generated_full, "t5_amr_enriched_results/test_results_amr_enriched_PARSED_full_0_6.txt")
    save_to_file(generated_output, "t5_amr_enriched_results/test_results_amr_enriched_PARSED_0_6.txt")


if __name__ == '__main__':
    # train_model(task_type=configs.TASK_TYPE_AMR_ENRICHED)
    generate_from_pretrained(
        task_type=configs.TASK_TYPE_AMR_ENRICHED,
        prefix=configs.TASK_AMR_ENRICHED_PREFIX,
        test_dataset_split=configs.TEST_SPLIT
    )

    # Single test triple
    # test_triple1 = {
    #     'gem_id': 'web_nlg_ru-validation-50',
    #     'gem_parent_id': 'web_nlg_ru-validation-50',
    #     'input': ['Perth | country | Australia'],
    #     'target': 'Перт находится в Австралии.',
    #     'references': ['Перт находится в Австралии.', 'Перт находится в Австралии.'],
    # }
    # test_triple2 = {
    #     'gem_id': 'web_nlg_ru-validation-100',
    #     'gem_parent_id': 'web_nlg_ru-validation-100',
    #     'input': ['Sheldon Moldoff | award | Inkpot Award'],
    #     'target': 'Премию Inkpot получил Шелдон Молдофф.',
    #     'references': ['Премию Inkpot получил Шелдон Молдофф.', 'Шелдон Молдофф получил премию Inkpot.'],
    # }
    # test_triple3 = {
    #     'gem_id': 'web_nlg_ru-validation-100',
    #     'gem_parent_id': 'web_nlg_ru-validation-100',
    #     'input': ['Israel | official language | Modern Hebrew'],
    # }
    #
    # generated = generate_one(test_triple3, tokenizer, finetuned_model, device)
    # print(generated)


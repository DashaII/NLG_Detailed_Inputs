#!/usr/bin/env python3

import configs
import datasets
import re
import unicodedata

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import pipeline

# ['Ab_Klink | almaMater | Erasmus_University_Rotterdam',
#  'Erasmus_University_Rotterdam | affiliation | Association_of_MBAs',
#  'Netherlands | currency | Euro',
#  'Ab_Klink | birthPlace | Stellendam',
#  'Ab_Klink | nationality | Netherlands']

# (1, 2)
# (2, 3)
# (4, 5)
# (1, 6)
# (1, 4)

# 1=Ab_Klink, 2=Erasmus_University_Rotterdam, 3=Association_of_MBAs, 4=Netherlands, 5=Euro, 6=Stellendam

# ['Ab_Klink | almaMater | Erasmus_University_Rotterdam',
#  'Erasmus_University_Rotterdam | affiliation | Association_of_MBAs',
#  'Ab_Klink | nationality | Netherlands']


# Born in the Netherlands, Ab Klink, attended Erasmus University Rotterdam which is affiliated to Association of MBAs.

# 4 1 2 3 -> r(1,4) (1,2) (2,3) -> v:4 < 1 > v: (2 > v:3)
# nationality: Netherlands < Ab_Klink > almaMater: [Erasmus_University_Rotterdam > affiliation: Association_of_MBAs]


def remove_diacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def transform_triple(triple):
    res_triple = triple.replace("_", " ")
    res_triple = res_triple.replace('"', '')
    res_triple = res_triple.replace("'", "")
    triple_split = res_triple.split("|")
    camel_to_space = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', triple_split[1])
    triple_split[1] = camel_to_space.lower()
    if triple_split[2][-2:] == '.0' and triple_split[2][:-2].strip().isdigit():
        triple_split[2] = triple_split[2][:-2]
    res_triple = (triple_split[0].strip(), triple_split[1].strip(), triple_split[2].strip())

    return res_triple


def transform_webnlg_data(dataset, size=None):
    # take just 2 first words of subj/object for further search in text
    def first_two_words(phrase):
        phrase_split = phrase.lower().split()
        if len(phrase_split) < 2:
            return remove_diacritics(phrase_split[0])
        # ignore second word: if there is a comma between the words OR the second word is in parentheses OR the
        # second word is "language"
        elif phrase_split[0][-1:] == "," or phrase_split[1][0] == "(" or phrase_split[1] == "language":
            return remove_diacritics(phrase_split[0].replace(',', ''))
        else:
            first_word = remove_diacritics(phrase_split[0])
            second_word = remove_diacritics(phrase_split[1].replace(',', ''))
            return f'{first_word} {second_word}'

    def create_schema(item):
        subj_obj_dict = {}
        triples = []
        for triple in item['input']:
            s, p, o = transform_triple(triple)
            sub_s = first_two_words(s)
            sub_o = first_two_words(o)
            if sub_s not in subj_obj_dict:
                subj_obj_dict[sub_s] = len(subj_obj_dict)+1
            if sub_o not in subj_obj_dict:
                subj_obj_dict[sub_o] = len(subj_obj_dict)+1
            triple_item = {"triple": (s, p, o), "schema": (subj_obj_dict[sub_s], subj_obj_dict[sub_o])}
            triples.append(triple_item)
        # unicode removes diacritics
        target = remove_diacritics(item["target"].lower().replace(",", ""))
        schema = {"triples": triples, "subj_obj_dict": subj_obj_dict, "target": target}
        return schema

    transformed = []
    for idx, item in enumerate(dataset):
        if size is not None and not (22290 <= idx < 22290 + size):
            continue
        transformed.append(create_schema(item))
    return transformed


def map_schema_to_target(dataset):
    DEBUG_LIST_ONE_WORD = []
    DEBUG_LIST_FIRST_FOUND_ONLY = []
    DEBUG_LIST_SECOND_FOUND_ONLY = []
    DEBUG_LIST_SECOND_BEFORE_FISRT = []
    DEBUG_LIST_OTHER = []

    for idx, item in enumerate(dataset):
        subj_obj_dict = item["subj_obj_dict"]
        target = item["target"]

        def find_position(k, targ):
            position = targ.find(k)
            if position != -1:
                return position, position+len(k)

            key_split = k.split()
            if len(key_split) == 1:
                position = targ.find(k[0:4])
                if position != -1:
                    return position, position + len(k)
                else:
                    DEBUG_LIST_ONE_WORD.append("key: "+k+" target: "+targ+" position: "+str(position))
                    return len(targ)+1, None

            first_pos = targ.find(key_split[0])
            second_pos = targ.find(key_split[1])
            if first_pos != -1 and second_pos != -1 and first_pos < second_pos:
                return first_pos, second_pos+len(key_split[1])
            elif second_pos == -1 and first_pos != -1:
                DEBUG_LIST_FIRST_FOUND_ONLY.append(
                    "key_split: " + k + " target: " + targ + " position1: " + str(first_pos) + " position2: " + str(second_pos))
            elif first_pos == -1 and second_pos != -1:
                DEBUG_LIST_SECOND_FOUND_ONLY.append(
                    "key_split: " + k + " target: " + targ + " position1: " + str(first_pos) + " position2: " + str(second_pos))
            elif first_pos > second_pos:
                DEBUG_LIST_SECOND_BEFORE_FISRT.append(
                    "key_split: " + k + " target: " + targ + " position1: " + str(first_pos) + " position2: " + str(second_pos))
            else:
                DEBUG_LIST_OTHER.append(
                    "key_split: " + k + " target: " + targ + " position1: " + str(first_pos) + " position2: " + str(second_pos))

            return len(targ) + 1, None

        # Sorting dictionary keys from longest to shortest ->
        # in cases when items with similar beginnings are present {"Aarhus Airport":1, "Aarhus":2}
        # the longest item should be searched first
        sorted_dict = sorted(subj_obj_dict.keys(), key=len, reverse=True)
        positions = {}
        for key in sorted_dict:
            start_position, end_position = find_position(key, target)
            # make a pair -> position in target sent : index of phrase
            positions[start_position] = subj_obj_dict[key]
            if end_position is not None:
                target = target.replace(key, "_"*(end_position-start_position))
        item["schema_positions"] = positions

    with open("data/DEBUG.txt", "w", encoding='utf-8') as file:
        file.write("DEBUG_LIST_ONE_WORD\n")
        file.write(str(len(DEBUG_LIST_ONE_WORD))+"\n")
        for item in DEBUG_LIST_ONE_WORD:
            file.write("%s\n" % item)

        file.write("\n\nDEBUG_LIST_FIRST_FOUND_ONLY\n")
        file.write(str(len(DEBUG_LIST_FIRST_FOUND_ONLY))+"\n")
        for item in DEBUG_LIST_FIRST_FOUND_ONLY:
            file.write("%s\n" % item)

        file.write("\n\nDEBUG_LIST_SECOND_FOUND_ONLY\n")
        file.write(str(len(DEBUG_LIST_SECOND_FOUND_ONLY))+"\n")
        for item in DEBUG_LIST_SECOND_FOUND_ONLY:
            file.write("%s\n" % item)

        file.write("\n\nDEBUG_LIST_SECOND_BEFORE_FISRT\n")
        file.write(str(len(DEBUG_LIST_SECOND_BEFORE_FISRT))+"\n")
        for item in DEBUG_LIST_SECOND_BEFORE_FISRT:
            file.write("%s\n" % item)

        file.write("\n\nDEBUG_LIST_OTHER\n")
        file.write(str(len(DEBUG_LIST_OTHER))+"\n")
        for item in DEBUG_LIST_OTHER:
            file.write("%s\n" % item)


def test():
    # t5_tokenizer = T5Tokenizer.from_pretrained(configs.HF_MODEL)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print("device type", device)
    # # model
    # t5_model = T5ForConditionalGeneration.from_pretrained(configs.HF_MODEL).to(device)
    # generate = pipeline(
    #     task="text-generation",
    #     model=t5_model,
    #     tokenizer=t5_tokenizer,
    #     device_map="auto"
    # )
    #
    # prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"
    #
    # sequences = generate(
    #     prompt,
    #     do_sample=True,
    #     max_new_tokens=100,
    #     # temperature=0.7,
    #     # top_k=50,
    #     # top_p=0.95,
    #     num_return_sequences=1,
    # )
    # print(sequences[0]['generated_text'])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=True,
    #                                              device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=True,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "My favourite condiment is"

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant",
         "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of "
                    "zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device type", device)
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
    output = tokenizer.batch_decode(generated_ids)[0]
    print(output)


def build_flat_plan(split, lang='en'):
    # test()
    dataset = datasets.load_dataset(path='GEM/web_nlg', name=lang, split=split)
    transformed = transform_webnlg_data(dataset, size=2000)
    map_schema_to_target(transformed)


if __name__ == '__main__':
    build_flat_plan(split=configs.TRAIN_SPLIT, lang=configs.LANG)


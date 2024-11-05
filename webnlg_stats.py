import datasets
import configs


def dataset_stats(split):
    data_en = datasets.load_dataset('GEM/web_nlg', 'en', split=split)

    category_counter = {}
    triples_counter = {}
    for item in data_en:
        triple_count = len(item["input"])
        item_cat = item["category"]
        category_counter[item_cat] = category_counter.get(item_cat, 0) + 1
        triples_counter[triple_count] = triples_counter.get(triple_count, 0) + 1

    category_counter = dict(sorted(category_counter.items()))
    triples_counter = dict(sorted(triples_counter.items()))

    print(split, len(data_en))
    print(category_counter)
    print(triples_counter)


def print_data_to_file(split: str, filename: str):
    data_en = datasets.load_dataset('GEM/web_nlg', 'en', split=split)
    with open(filename, "w", encoding='utf-8') as file:
        for item in data_en:
            # file.write("%s\n" % str(item['input']))
            file.write("%s\n" % item['target'])


if __name__ == '__main__':
    # dataset_stats(configs.TRAIN_SPLIT)
    # dataset_stats(configs.VALID_SPLIT)
    # dataset_stats(configs.TEST_SPLIT)

    print_data_to_file(configs.TEST_SPLIT, "data/webnlg_data_targets_test.txt")

    # for idx, item in enumerate(data_en):
    # with open("webnl_en_data_test.txt", "w", encoding='utf-8') as file:
    #     for i, item in enumerate(data_en):
    #         file.write("%s\n" % item["input"])

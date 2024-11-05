#!/usr/bin/env python3

import sacrebleu
from bleurt import score
import configs
import datasets
import numpy as np
from tqdm import tqdm
import warnings
from datasets import load_metric
from evaluate import load

warnings.filterwarnings("ignore")

TEST_RESULTS_FILES = [
    't5_amr_enriched_results/test_results_amr_enriched_PARSED_0_6.txt'
]
VALID_RESULT_FILES = [
    't5_results/dev_results_generated_4_3.txt',
    't5_results/dev_results_generated_5_15.txt',
    't5_results/dev_results_generated_5_6.txt',
    't5_results/dev_results_generated_6_6.txt',
]
WEBNLG_RESULT_FILES = [
    'WebNLG_2020_results/Amazon_AI_(Shanghai)/primary.en',
    'WebNLG_2020_results/FBConvAI/primary.en',
    'WebNLG_2020_results/cuni-ufal/primary.en',
    'WebNLG_2020_results/RALI/primary.en',
    'WebNLG_2020_results/Baseline-FORGE2020/primary.en',
    'WebNLG_2020_results/Baseline-FORGE2017/primary.en'
]


def get_category_and_triple_count(dataset_split: str):
    dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=dataset_split)
    # stats = [(item['category'], len(item['input'])) for item in dataset]
    stats = []
    test_cat_idx = []
    dev_cat_idx = []
    for i, item in enumerate(dataset):
        if i < len(dataset):
            stats.append((item['category'], len(item['input'])))
            if item['category'] in ('MusicalWork', 'Scientist', 'Film'):
                test_cat_idx.append(i)
            else:
                dev_cat_idx.append(i)
        else:
            break
    return stats, test_cat_idx, dev_cat_idx


def read_from_file(filename):
    with open(filename, mode="r", encoding="utf-8") as file:
        result = [line.strip() for line in file]
    return result


def get_reference(data):
    res = [item['references'] for item in data]
    max_len = max(len(sub_array) for sub_array in res)
    res_padded = np.array([sub_array + [''] * (max_len - len(sub_array)) for sub_array in res])
    res_transposed = res_padded.T.tolist()
    return res_transposed


def get_scores(dataset_split: str, result_files_names: list):
    """
    Get dataset split parameter (valid or test) and list of files names with generated results.
    Prints BLEU, CHRF2, BLEURT and METEOR scores.
    """
    test_dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=dataset_split)
    references = get_reference(test_dataset)

    results = []
    for file_name in result_files_names:
        results.append(read_from_file(file_name))

    # BLEU and CHRF
    bleu = sacrebleu.metrics.BLEU()
    chrf = sacrebleu.metrics.CHRF()

    # BLEURT
    # checkpoint = "bleurt/BLEURT-20-D12"
    checkpoint = "bleurt/bleurt-base-128"
    bleurt_scorer = score.BleurtScorer(checkpoint=checkpoint)
    # bleurt_scorer = score.LengthBatchingBleurtScorer(checkpoint=checkpoint)

    # METEOR
    meteor = load('meteor')

    for i, result in enumerate(results):
        # Compute BLEU and CHRF scores
        bleu_score = bleu.corpus_score(hypotheses=result, references=references)
        chrf_score = chrf.corpus_score(hypotheses=result, references=references)

        # Compute the BLEURT
        bluert_scores_list = []
        for ref in tqdm(references):
            bluert_scores_list.append(bleurt_scorer.score(references=ref, candidates=result, batch_size=200))
        bluert_scores_list = np.array(bluert_scores_list)
        bluert_scores_list = bluert_scores_list.max(axis=0)
        avg_bluert = np.average(bluert_scores_list)

        # Compute the METEOR score
        meteor_ref = [item['references'] for item in test_dataset]
        # calculates the average
        meteor_results = meteor.compute(predictions=result, references=meteor_ref)
        meteor_score = meteor_results["meteor"]

        print("\n", result_files_names[i], ">")
        print("BLEU score", round(bleu_score.score, 2))
        print("CHRF score", chrf_score)
        print("BLEURT score", round(avg_bluert, 4))
        print("METEOR score", round(meteor_score, 4))


def get_bleurt_scores_by_categories(dataset_split: str, result_files_names: list):
    """
    Get dataset split parameter (valid or test) and list of files names with generated results.
    Prints BLEURT scores for dev only categories, test only categories (seen in test dataset only) and average score.
    """
    test_dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=dataset_split)
    references = get_reference(test_dataset)

    results = []
    for file_name in result_files_names:
        results.append(read_from_file(file_name))

    # BLEURT
    # checkpoint = "bleurt/BLEURT-20-D12"
    checkpoint = "bleurt/bleurt-base-128"
    bleurt_scorer = score.BleurtScorer(checkpoint=checkpoint)
    # bleurt_scorer = score.LengthBatchingBleurtScorer(checkpoint=checkpoint)

    stats, test_cat_idx, dev_cat_idx = get_category_and_triple_count(dataset_split)

    for i, result in enumerate(results):
        bluert_scores_list = []
        for ref in tqdm(references):
            bluert_scores_list.append(bleurt_scorer.score(references=ref, candidates=result, batch_size=100))
        bluert_scores_list = np.array(bluert_scores_list)
        bluert_scores_list = bluert_scores_list.max(axis=0)

        avg_bluert = np.average(bluert_scores_list)

        print("\n", result_files_names[i], ">")
        print("BLEURT score", round(avg_bluert, 4))


if __name__ == '__main__':
    # get_scores(configs.VALID_SPLIT, VALID_RESULT_FILES)
    # get_scores(configs.TEST_SPLIT, TEST_RESULTS_FILES)
    get_scores(configs.TEST_SPLIT, TEST_RESULTS_FILES)

    # BLEURT
    # cand = ['Municipal Coaracy da Mata Fonseca is located in Arapiraca and is the home ground of the Agremiacao Sportiva Arapiraquense. The club play in the Campeonato Brasileiro Série C league in Brazil and the nickname of the player is Alvinegro.']
    # ref1 = ['Estádio Municipal Coaracy da Mata Fonseca is the name of the ground of Agremiação Sportiva Arapiraquense in Arapiraca. Agremiação Sportiva Arapiraquense, nicknamed "Alvinegro", lay in the Campeonato Brasileiro Série C league from Brazil.']
    # ref2 = ['Estádio Municipal Coaracy da Mata Fonseca is the name of the ground of Agremiação Sportiva Arapiraquense in Arapiraca. Alvinegro, the nickname of Agremiação Sportiva Arapiraquense, play in the Campeonato Brasileiro Série C league from Brazil.']
    #
    # checkpoint = "bleurt/BLEURT-20"
    # scorer = score.BleurtScorer(checkpoint)
    # score1 = scorer.score(references=ref1, candidates=cand)
    # score2 = scorer.score(references=ref2, candidates=cand)
    #
    # print(score1, score2)

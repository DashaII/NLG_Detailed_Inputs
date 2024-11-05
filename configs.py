#!/usr/bin/env python3

import os

SPECIAL_TOKENS = ['<|triple|>', '<|target|>', '<|endoftext|>', '<|structure|>']

# T5 models need a slightly higher learning rate than the default one set in the Trainer when using the AdamW optimizer.
# Typically, 1e-4 and 3e-4 work well for most problems (classification, summarization, translation, question answering,
# question generation). Note that T5 was pre-trained using the AdaFactor optimizer.
LEARNING_RATE = 1e-4
EPOCHS = 2
BATCH_SIZE = 5

# Set None for full dataset
DATASET_SIZE = None

# google-t5/t5-small | google-t5/t5-base | google-t5/t5-large | google-t5/t5-3b | google-t5/t5-11b
# google/flan-t5-small | google/flan-t5-base | google/flan-t5-large | google/flan-t5-3b | google/flan-t5-11b
HF_MODEL = "google-t5/t5-small"
TASK_BASELINE_PREFIX = (
    "Each triple consists of a subject, a predicate, and an object. Identify the relationship described by "
    "the predicate between the subject and the object. Combine the subject, predicate, and object into a "
    "grammatically correct sentence. If multiple triples are provided, integrate them into a single "
    "coherent text.")
TASK_AMR_PREFIX = "Transform triples into sentences of target structure."
TASK_AMR_ENRICHED_PREFIX = "Transform triples and sentence structure into coherent text."

# en OR ru
LANG = 'en'

# train, validation, test,
# challenge_train_sample, challenge_validation_sample
# challenge_test_scramble, challenge_test_numbers
TRAIN_SPLIT = 'train'
VALID_SPLIT = 'validation'
TEST_SPLIT = 'test'

MODEL_NAME = "model_state.pt"
MODEL_PATH = os.path.join(os.curdir, 't5_amr_enriched_models', 't5_model_amr_enriched_0_6', 't5_webnlg_epoch6')
MODEL_PATH_FOLDER = 't5_model_amr_enriched_TEST_DELETE'

RESULTS_FILE = 'results/results_generated_14_9.txt'
FULL_RESULTS_FILE = 'results/results_generated_full_14_13.txt'

TASK_TYPE_BASELINE = "baseline"
TASK_TYPE_AMR = "amr"
TASK_TYPE_AMR_ENRICHED = "amr_enriched"

AMR_PARSER_SIMPLIFIED = "data/flat_amr_parsing_results/amr_parser_simplified_"
AMR_GENERATED_SIMPLIFIED = "data/flat_amr_parsing_results/amr_parser_simplified_"
# AMR_GENERATED_SIMPLIFIED = "data/flat_amr_generated_results/results_generated_2_10_"

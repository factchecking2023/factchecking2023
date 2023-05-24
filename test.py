"""
Tests model.

"""
import os
import json
from argparse import ArgumentParser, Namespace

import pandas as pd
import time
import yaml
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from classifier import Classifier
from utils import my_read_csv, remove_dup_space, get_limit_text, evaluate

def load_model_from_experiment(experiment_folder: str):
    """Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    hparams_file = experiment_folder + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]
    model = Classifier.load_from_checkpoint(
        checkpoint_path, hparams=Namespace(**hparams)
    )
    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Path to the experiment folder.",
    )
    parser.add_argument(
        "--test_csv",
        default='',
        required=False,
        type=str,
        help="Path to the test data.",
    )
    parser.add_argument(
        "--store_predictions", "-o",
        default='predictions',
        required=False,
        type=str,
        help="Path to store predictions.",
    )
    parser.add_argument(
        "--eval_only",
        action='store_true',
        help="Whether to evaluate only",
    )
    parser.add_argument(
        "--claim_id",
        default=-1,
        required=False,
        type=int,
        help="claim_id",
    )
    parser.add_argument(
        "--use_cache",
        action='store_true',
        help="Whether to use cache",
    )
    hparams = parser.parse_args()

    pred_path = os.path.join(hparams.experiment, "predictions2.json")

    if hparams.eval_only and os.path.exists(pred_path):
        print("pred_path:", pred_path)
        evaluate(hparams, pred_path)
        print("done")
        exit()

    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment)
    
    df = my_read_csv(hparams.test_csv)
    testset = df.to_dict("records")

    print("pred_path:", pred_path)
    start_time = time.time()
    predictions = [
        model.predict(sample)
        for sample in tqdm(testset, desc="Testing on {}".format(hparams.test_csv))
    ]
    end_time = time.time()
    test_time_seconds = end_time - start_time
    print(f"=== Test time: {test_time_seconds:.2f} seconds")
    minutes, seconds = divmod(test_time_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print("=== Test time (h-m-s): {} h {} m {:.2f} s".format(int(hours), int(minutes), seconds))

    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print ("saving predictions at: {}".format(hparams.store_predictions))
    evaluate(hparams, pred_path)
    print("done")
    


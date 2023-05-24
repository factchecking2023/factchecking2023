"""
Runs a model on a single node across N-gpus.

"""
import argparse
import os
from datetime import datetime
import time
from argparse import ArgumentParser, Namespace

from classifier import Classifier
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchnlp.random import set_seed
import yaml

from classifier import DataModule
from utils import my_read_csv, evaluate
from tqdm import tqdm
import json

class TimeCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            end_time = time.time()
            training_time_seconds = end_time - self.start_time

            # 打印训练时间
            print(f"=== Training time: {training_time_seconds:.2f} seconds")

            # 将秒转换为时、分、秒的形式
            minutes, seconds = divmod(training_time_seconds, 60)
            hours, minutes = divmod(minutes, 60)

            # 打印结果
            print("=== Training time (h-m-s): {} h {} m {:.2f} s".format(int(hours), int(minutes), seconds))

def load_model_from_experiment(experiment_folder: str, is_finetune=False):
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
    if not is_finetune:
        model.eval()
        model.freeze()
    return model

def start_training(hparams) -> None:
    print("==start_training")
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------

    if hparams.is_finetune:
        model = load_model_from_experiment(hparams.experiment, is_finetune=True)
    else:
        model = Classifier(hparams)

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    # ------------------------
    # 3 INIT LOGGERS
    # ------------------------
    # Tensorboard Callback
    tb_logger = TensorBoardLogger(
        save_dir=hparams.save_dir,
        version=hparams.version,
        name="",
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join(
        hparams.save_dir,
        hparams.version,
        "checkpoints",
    )

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=ckpt_path,
    #     save_top_k=hparams.save_top_k,
    #     verbose=True,
    #     monitor=hparams.monitor,
    #     period=1,
    #     mode=hparams.metric_mode,
    #     save_weights_only=True,
    # )

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        # checkpoint_callback=True,
        callbacks=[TimeCallback()],
        gradient_clip_val=1.0,
        gpus=hparams.gpus,
        # log_gpu_memory="all",
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
        # distributed_backend="dp",
        # limit_val_batches=0,  # 设置为 0 或 False 可以跳过验证过程
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------
    # trainer.fit(model, model.data)

    data = DataModule(model, hparams)
    trainer.fit(model, data)
    cmd = "python test.py --experiment {}{} --eval_only --test_csv {}".format(hparams.save_dir, hparams.version, hparams.test_csv)
    print("*" * 50)
    print("training finished, evaluating... Or you can run the following command manually to evaluate again:")
    print("    ", cmd)
    print("*" * 50)
    # os.system(cmd)


    log_dir = os.path.join("data/logs/", hparams.version)
    try:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        cmd = "cp {}{}/events.out.* {}".format(hparams.save_dir, hparams.version, log_dir)
        os.system(cmd)
        print("=== saved log file to {}".format(log_dir))
    except Exception as e:
        print("=== ERROR: ", str(e))

    model.eval()
    model.freeze()
    pred_path = os.path.join(hparams.save_dir, hparams.version, "predictions.json")
    df = my_read_csv(hparams.test_csv)
    testset = df.to_dict("records")
    # predictions = []
    # for sample in tqdm(testset, desc="Testing on {}".format(hparams.test_csv)):
    #     out = model.predict(sample)
    #     exit()

    print("pred_path:", pred_path)
    predictions = [
        model.predict(sample)
        for sample in tqdm(testset, desc="Testing on {}".format(hparams.test_csv))
    ]
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print("saving predictions at: {}".format(pred_path))
    evaluate(hparams, pred_path)



if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="early_stop_on", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=2,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=1,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    parser.add_argument(
        "--attn_type",
        default=0,
        type=int,
        help="Attention layer types.",
    )
    parser.add_argument(
        "--save_dir",
        default='',
        required=True,
        type=str,
        help="File path for saving models and outputting results",
    )

    # Fine tuning
    parser.add_argument(
        "--is_finetune",
        action='store_true',
        help="Whether to finetune",
    )
    parser.add_argument(
        "--experiment",
        default='',
        required=False,
        type=str,
        help="Fine-tuning based on the specified experiment.",
    )

    # each LightningModule defines arguments relevant to it
    parser = Classifier.add_model_specific_args(parser)

    hparams = parser.parse_args()
    hparams.version = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")


    # ---------------------
    # RUN TRAINING
    # ---------------------
    if hparams.is_finetune:
        if hparams.experiment == '':
            print("Please provide the --experiment parameter")
            exit()
    start_training(hparams)

    print("done")

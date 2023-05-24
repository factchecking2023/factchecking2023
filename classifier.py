# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from transformers import AutoModel
from transformers import AutoConfig

from tokenizer import Tokenizer
from utils import mask_fill, my_read_csv
# from mha import *
from torch.nn.functional import normalize


class DataModule(pl.LightningDataModule):
    def __init__(self, classifier_instance, hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.classifier = classifier_instance
        # Label Encoder
        self.label_encoder = LabelEncoder(
            my_read_csv(self.hparams.train_csv).label.astype(str).unique().tolist(),
            reserved_labels=[],
        )
        self.label_encoder.unknown_index = None

    def read_csv(self, path: str) -> list:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        print("===reading:", path)
        df = my_read_csv(path)
        return df.to_dict("records")

    def train_dataloader(self) -> DataLoader:
        """Function that loads the train set."""
        self._train_dataset = self.read_csv(self.hparams.train_csv)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.classifier.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        self._dev_dataset = self.read_csv(self.hparams.dev_csv)
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.classifier.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Function that loads the test set."""
        self._test_dataset = self.read_csv(self.hparams.test_csv)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.classifier.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

class Classifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.

    :param hparams: ArgumentParser containing the hyperparameters.
    """
    def __init__(self, hparams: Namespace) -> None:
        super(Classifier, self).__init__()
        # self.hparams = hparams
        # self.hparams.update(hparams)
        self.save_hyperparameters(hparams)
        self.batch_size = hparams.batch_size

        # Build Data module
        self.data = DataModule(self, hparams)

        # build model
        self.attn_type = hparams.attn_type
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        self.bert = AutoModel.from_pretrained(
            self.hparams.encoder_model, output_hidden_states=True
        )

        # set the number of features our encoder model will return...
        self.encoder_features = self.bert.config.hidden_size

        # Tokenizer
        self.tokenizer = Tokenizer(self.hparams.encoder_model)

        print("self.data.label_encoder.vocab_size: {}".format(self.data.label_encoder.vocab_size))

        self.mha_heads_num = 4
        self.mha = nn.MultiheadAttention(self.encoder_features, self.mha_heads_num)

        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features * 2, self.encoder_features * 4),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 4, self.encoder_features * 4),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 4, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),
        )

    def __build_loss(self):
        """Initializes the loss function/s."""
        self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True




    def embedding_layer(self, tokens, lengths):
        """
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        """
        tokens = tokens[:, : lengths.max()]
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        word_embeddings = self.bert(tokens, mask)[0] # [batch_size x src_seq_len x dims]

        # Average Pooling
        # Fills elements of self tensor with value (0.0 below) where mask is True.
        # The shape of mask must be broadcastable with the shape of the underlying tensor.
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        ) # [batch_size x src_seq_len x dims]
        return word_embeddings, mask


    def forward(self, t1, l1, t2, l2, t3, l3, t4, l4):
        """Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        # print("embeddings.size()", embeddings.size()) # (batch_size, sequence_length, embed_dim)
        # print("padding_mask.size()", padding_mask.size()) # (batch_size, sequence_length)

        embeddings1, padding_mask1 = self.embedding_layer(t1, l1)
        padding_mask1 = ~padding_mask1 # (batch_size, sequence_length)
        embeddings1 = embeddings1.transpose(0, 1)  # (sequence_length, batch_size, embed_dim)

        embeddings2, padding_mask2 = self.embedding_layer(t2, l2)
        padding_mask2 = ~padding_mask2 # (batch_size, sequence_length)
        embeddings2 = embeddings2.transpose(0, 1)  # (sequence_length, batch_size, embed_dim)

        f1 = torch.cat((embeddings1, embeddings2), dim=0)
        m1 = torch.cat((padding_mask1, padding_mask2), dim=1)

        embeddings3, padding_mask3 = self.embedding_layer(t3, l3)
        padding_mask3 = ~padding_mask3
        embeddings3 = embeddings3.transpose(0, 1)  # (sequence_length, batch_size, embed_dim)

        embeddings4, padding_mask4 = self.embedding_layer(t4, l4)
        padding_mask4 = ~padding_mask4
        embeddings4 = embeddings4.transpose(0, 1)  # (sequence_length, batch_size, embed_dim)

        f2 = torch.cat((embeddings3, embeddings4), dim=0)
        m2 = torch.cat((padding_mask3, padding_mask4), dim=1)

        o1, _ = self.mha(f1, f1, f1, key_padding_mask=m1)
        # o2, _ = self.mha(f1, f2, f2, key_padding_mask=m2)
        o3, _ = self.mha(f2, f2, f2, key_padding_mask=m2)

        x = torch.cat((o1[-1], o3[-1]), dim=1)

        del embeddings1, padding_mask1, embeddings2, padding_mask2, embeddings3, padding_mask3, embeddings4, padding_mask4
        del f1, m1, f2, m2, o1, o3
        torch.cuda.empty_cache()

        return {"logits": self.classification_head(x), "attn": None}


    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        t1, l1 = self.tokenizer.batch_encode(sample["claim"])
        t2, l2 = self.tokenizer.batch_encode(sample["main_text"])
        t3, l3 = self.tokenizer.batch_encode(sample["claim_entities_desc"])
        t4, l4 = self.tokenizer.batch_encode(sample["text_entities_desc"])

        inputs = {
            "t1": t1, "l1": l1,
            "t2": t2, "l2": l2,
            "t3": t3, "l3": l3,
            "t4": t4, "l4": l4,
        }

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.data.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def predict(self, sample: dict):
        """Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            # logits = model_out["logits"].numpy()
            m = nn.Softmax(dim=1)
            logits = m(model_out["logits"]).numpy()

            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]
        return sample

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        loss_val = loss_val.unsqueeze(0)
        # print("type of loss_val:", loss_val)

        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True)

        output = OrderedDict({"loss": loss_val})
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        loss_val = loss_val.unsqueeze(0)
        val_acc = val_acc.unsqueeze(0)
        # self.log("val_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("val_accuracy", val_acc, on_step=True, on_epoch=True, prog_bar=True)

        output = OrderedDict(
            {
                "val_loss": loss_val,
                "val_acc": val_acc,
            }
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs: list) -> dict:
        """Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_train_epoch_end(self):
        """Pytorch lightning hook"""
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()
    def on_validation_epoch_end(self):
        """Pytorch lightning hook"""
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()
    def on_test_epoch_end(self):
        """Pytorch lightning hook"""
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Parser for Estimator specific arguments/hyperparameters.
        :param parser: argparse.ArgumentParser

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default="bert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        parser.add_argument(
            "--train_csv",
            required=True,
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            required=True,
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            required=True,
            type=str,
            help="Path to the file containing the test data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=2,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser

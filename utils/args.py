import argparse
import dataclasses
from typing import Optional
from dataclasses import dataclass


@dataclass
class PreprocessArgs:
    """
    Arguments for preprocess
    """

    dataset: str
    dataset_type: str
    llm: str
    bert: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    tensor_parallel_size: int
    skip_inference: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Path of the dataset",
        )
        parser.add_argument(
            "--dataset_type",
            type=str,
            required=True,
            help="Type of the dataset, e.g. 'shareGPT'",
        )
        parser.add_argument(
            "--llm",
            type=str,
            required=True,
            help="Name or path of the LLM",
        )
        parser.add_argument(
            "--bert",
            type=str,
            required=True,
            help="Name or path of the pretrained bert model",
        )

        # inference args
        parser.add_argument(
            "--max_tokens",
            type=int,
            default=1e5,
            help="Maximum number of tokens to generate per output sequence.",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1,
            help="Temperature of inference",
        )
        parser.add_argument(
            "--top_p",
            type=float,
            default=1,
            help="Top_p of inference",
        )
        parser.add_argument(
            "--top_k",
            type=int,
            default=-1,
            help="Top_k of inference",
        )
        parser.add_argument(
            "--tensor_parallel_size",
            type=int,
            default=1,
            help="Tensor parallel size of inference",
        )
        parser.add_argument(
            "--skip_inference",
            action="store_true",
            help="Enable preprocess to skip inference",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        preprocess_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return preprocess_args


@dataclass
class TrainArgs:
    """
    Arguments for model training
    """

    dataset: str
    dataset_type: str
    llm: str
    bert: str
    epoch: int
    train_bs: int
    validate_bs: int
    bert_lr: float
    lr: float
    hidden_dim: int

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Path of the dataset",
        )
        parser.add_argument(
            "--dataset_type",
            type=str,
            required=True,
            help="Type of the dataset, e.g. 'shareGPT'",
        )
        parser.add_argument(
            "--llm",
            type=str,
            required=True,
            help="Name or path of the LLM",
        )
        parser.add_argument(
            "--bert",
            type=str,
            required=True,
            help="Name or path of the pretrained bert model",
        )
        # train
        parser.add_argument(
            "--epoch",
            type=int,
            default=4,
            help="Number of epochs",
        )
        parser.add_argument(
            "--train_bs",
            type=int,
            default=32,
            help="Batch size of training",
        )
        parser.add_argument(
            "--validate_bs",
            type=int,
            default=32,
            help="Batch size of validation",
        )
        parser.add_argument(
            "--bert_lr",
            type=float,
            default=5e-5,
            help="Learning rate of bert",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="Learning rate of the model",
        )
        # model
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=128,
            help="Hidden dimension of the model",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        preprocess_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return preprocess_args


@dataclass
class TestArgs:
    """
    Arguments for model testing
    """

    dataset: str
    dataset_type: str
    llm: str
    bert: str

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Path of the dataset",
        )
        parser.add_argument(
            "--dataset_type",
            type=str,
            required=True,
            help="Type of the dataset, e.g. 'shareGPT'",
        )
        parser.add_argument(
            "--llm",
            type=str,
            required=True,
            help="Name or path of the LLM",
        )
        parser.add_argument(
            "--bert",
            type=str,
            required=True,
            help="Name or path of the pretrained bert model",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        preprocess_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return preprocess_args

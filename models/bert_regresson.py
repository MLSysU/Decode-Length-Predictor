import os
import torch
import torch.nn as nn
from transformers import BertModel


class BertRegressionModel(nn.Module):
    def __init__(self, bert: BertModel, cls_thresholds: list, hidden_dim: int):
        super().__init__()
        self.ceil_thresholds = torch.tensor(cls_thresholds)
        self.floor_thresholds = torch.tensor([0] + cls_thresholds[:-1])
        self.bert = bert
        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output

    def get_cls(self, input: torch.Tensor) -> torch.Tensor:
        """
        Return the classes of values based on the thresholds.
        """
        around_tensor = input.round()
        cls_tensor = around_tensor.clip(0, len(self.ceil_thresholds)).long()
        return cls_tensor

    def get_interval(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the interval of the input tensor based on the thresholds.
        """
        cls_tensor = self.get_cls(input)
        floor_tensor = torch.index_select(self.floor_thresholds, 0, cls_tensor)
        ceil_tensor = torch.index_select(self.ceil_thresholds, 0, cls_tensor)
        return floor_tensor, ceil_tensor

    def save_pretrained(self, save_directory: str):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the `from_pretrained()` class method.
        """
        state_dict = {
            "bert_config": self.bert.config,
            "cls_thresholds": self.ceil_thresholds.tolist(),
            "hidden_dim": self.cls.out_features,
            "state_dict": self.state_dict(),
        }
        os.makedirs(save_directory, exist_ok=True)
        torch.save(state_dict, os.path.join(save_directory, "model.pth"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "BertRegressionModel":
        """
        Instantiate a BertRegressionModel from a pre-trained model.
        """
        state_dict = torch.load(os.path.join(pretrained_model_name_or_path, "model.pth"), weights_only=False)
        bert = BertModel(state_dict["bert_config"])
        model = cls(bert, state_dict["cls_thresholds"], state_dict["hidden_dim"])
        model.load_state_dict(state_dict["state_dict"])
        return model

    @classmethod
    def from_bert(cls, bert_model_name_or_path: str, cls_thresholds: list, hidden_dim: int = 128) -> "BertRegressionModel":
        """
        Instantiate a BertRegressionModel from a pre-trained Bert model for training.
        """
        bert = BertModel.from_pretrained(bert_model_name_or_path)
        return cls(bert, cls_thresholds, hidden_dim)

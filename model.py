import torch.nn as nn
from transformers import BertModel, AutoConfig


class BertClassificationModel(nn.Module):
    def __init__(self, model_name, hidden_dim, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(self.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.logsoftmax(self.fc2(output))
        return output


class BertRegressionModel(nn.Module):
    def __init__(self, model_name, hidden_dim):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        # Fix the weights of the pretrained model
        for param in self.bert.parameters():
            param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(self.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output

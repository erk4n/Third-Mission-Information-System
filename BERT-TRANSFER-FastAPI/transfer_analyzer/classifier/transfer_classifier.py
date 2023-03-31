import json
from torch import nn
from transformers import BertModel

with open("config.json") as json_file:
    config = json.load(json_file)


class BertClassifier(nn.Module):
    
    def __init__(self, n_classes):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-german-cased', output_attentions=True)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)[0], self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)[1]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.softmax(linear_output)

        return final_layer, self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)[2]
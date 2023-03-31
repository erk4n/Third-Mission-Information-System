import torch
from transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
labels = {'interne Aktion':0,
          'Akteur':1,
          'Medium / Plattform':2,
          'Gründung':3,
          'externe Aktion':4
          }

class Dataset(torch.utils.data.Dataset):
    
    labels = {'interne Aktion':0,
          'Akteur':1,
          'Medium / Plattform':2,
          'Gründung':3,
          'externe Aktion':4
          }
    def __init__(self, df):

        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['Articledetail']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
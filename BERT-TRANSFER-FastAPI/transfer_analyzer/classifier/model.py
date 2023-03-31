
import json

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer

from .transfer_classifier import BertClassifier
from .transfer_dataset import Dataset

from torch.optim import Adam
from tqdm import tqdm

import pandas as pd
import re
import numpy as np

from .db import db  
import string
import spacy

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])

        classifier = BertClassifier(len(config["CLASS_NAMES"]))
        classifier.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        
        encoded_text = self.tokenizer(text, truncation=True, return_tensors="pt")

        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        model_output = self.classifier(input_ids,attention_mask)
#berechnung attention
        attention_matrix = model_output[1]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0]) 
        attention = self.get_attention(tokens, attention_matrix, text)

        probabilities = model_output[0]
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.detach().numpy()
        print(probabilities)
        return (
            config["CLASS_NAMES"][predicted_class],
            confidence,
            dict(zip(config["CLASS_NAMES"], probabilities[0])),
            attention
        )
        
    def get_attention(self, tokens, matrix, text):
        raw = torch.cat(matrix).sum(0).sum(0).sum(0).tolist()
        list_of_tokens_to_remove = [i for i in tokens if re.findall(r"##+", i)]
        list_of_tokens_to_remove = list_of_tokens_to_remove + [i for i in tokens if re.findall(r"[SEP]", i)]
        list_of_tokens_to_remove = list_of_tokens_to_remove + [i for i in tokens if re.findall(r"[CLS]", i)]

        print('1', len(list_of_tokens_to_remove))
        #for c in string.punctuation:
        #    print(c)
        #    list_of_tokens_to_remove = list_of_tokens_to_remove + [i for i in tokens if re.findall(c, i)]
        print('list_of_tokens_to_remove', list_of_tokens_to_remove)


        indexes = []
        print('token', tokens, len(tokens))
        for element in list_of_tokens_to_remove:
            indexes.append(self.find_element_in_list(element, tokens))
        print('indexes', indexes, len(indexes))
        indexes.sort(reverse=True)

        indexes = [x - 1 for x in indexes]
        for i in indexes:
            raw.pop(i)
        norm = [float(i)/sum(raw) for i in raw]
        max_val =sorted(range(len(norm)), key=lambda i: norm[i])[-10:]
        print(norm)

#        norm = self.softmax(raw)
#        print(norm.tolist())
        print('norm', norm, len(norm))

        pretok_sent = ""
        i = 0
        for tok in tokens:
            if tok.startswith("##"):
                pretok_sent += tok[2:]
            else:
                pretok_sent += " " + tok
                i =i+1

        pretok_sent = pretok_sent[1:]
        print(pretok_sent, i)
        return norm
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def find_element_in_list(self, element, list_element):
        try:
            index_element = list_element.index(element)
            return index_element
        except ValueError:
            return None

        
    def train_conf(self):
    
        #Data
        docids = self.get_db().collection.find({"annotated?" : { "$eq" : True}})
        res = [app for app in docids]
        print("data", docids)
                
        df = pd.DataFrame(res)
        df = df.reset_index()
        df_totrain = pd.DataFrame()

        for index, row in df.iterrows():
            if(row['ki-prediction'] != row['label']):
                if(max(row['prob']) < 0.99):
                    df_totrain = df_totrain.append(row, ignore_index = True)
                    print("dict", dict)
                    print("row", row)
    

        df_totrain = df_totrain.reset_index()    
        print("df_totrain", df_totrain)
        model = self.classifier
        df_train, df_val, df_test = np.split(df_totrain.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

        print("Trainingslaenge", len(df_train),len(df_val), len(df_test))
        #Hyperparameter
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        
        self.train(self.classifier, df_train, df_val, learning_rate, batch_size, 1)
        
        PATH = "entire_model.pt"
        torch.save(self.classifier.state_dict(), PATH)
        print("Hier gespeichert: ", PATH)
                
            
    def get_db(self):
        return db
    
    def train(self, model, train_data, val_data, learning_rate, batch_size, epochs):
        
        continue_training = True
        last_loss_values = []

        train, val = Dataset(train_data), Dataset(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr= learning_rate)

        if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()


        for epoch_num in range(epochs):
      
            if(continue_training):

                total_acc_train = 0
                total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)[0]
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)[0]

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
 

model = Model()


def get_model():
    return model

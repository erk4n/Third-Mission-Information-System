# Third-Mission-Information-System

This repository contains the implementation of a Third Mission Information System for capturing, analyzing and visualizing of activities concerning the knowledge and technology transfer of universities.

To make the programme operational, several components have to be installed. For that, this repository needs to be cloned first. For the three components '', do following seperatly in each terminal: 

1. open terminal and cd into the project.
2. activate virtual environment and install requirments

## Prerequisites
- python 3.x installed
- pip package installer installed

## Database
First, go to https://account.mongodb.com/account/login and create an mongoDB database account. Create the database called 'transfer' and copy the connection string.

## Scraping tool

1. cd into directory /transferactivity-Scrapy
2. enter scrapyrt

## Machine Learning

1. cd into directory /BERT-TRANSFER-FastAPI
2. enter uvicorn transfer_analyzer.api:app

## Information System

1. open project in IDE of choice and search for CONNECTION string. Paste the connection string copied from your mongoDB account.
2. cd into directory /BERT-TRANSFER-FastAPI
3. enter python app.py

from __future__ import unicode_literals
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import json
import requests
import pandas as pd
from datetime import datetime
import db
from prediction import *
from flask import jsonify
from flask_wtf import FlaskForm
from random import randint
from bson.objectid import ObjectId
from flask_wtf.csrf import CSRFProtect
from flask import Flask, request, Response, render_template, session, redirect, url_for
import w3lib.html
from flask import Markup
from collections import Counter

from flask_pymongo import pymongo
from wtforms import TextField, TextAreaField, SelectField, SubmitField, RadioField, validators
from wtforms.validators import InputRequired, Length
from domain.Article import Article
from flask_crontab import Crontab
import spacy

app = Flask(__name__)
crontab = Crontab(app)

app.secret_key = 'SoMeThInG000SeCrEt'
csrf = CSRFProtect(app)

@app.route('/activities/pie-chart')
def google_pie_chart():
    docids = get_db().collection.find({"annotated?" : { "$eq" : True}})
    res = [app['label'] for app in docids]
    data1 = {'Task' : 'Hours per Day'}
    c = Counter(res)
    c['Task'] = 'Hours per Day'
    data1.update(c)

    datums = get_db().collection.find({"Datum" : { "$exists" : True}})
    datums = [app['Datum'].split(' ',1)[1] for app in datums]

    data2={'Name':'Anzahl'}
    d = Counter(datums)
    d['Name'] = 'Anzahl'
    data2.update(d)
    
    authors = get_db().collection.find({"Autor" : { "$exists" : True}})
    authors = [author['Autor'] for author in authors]
    data3={}
    e = Counter(authors)#.keys() #liefert distinct Namen zurück
    data3.update(e)
    return render_template('pie-chart.html', data1=data1, data2=data2, data3=data3)

@app.route('/main')
def main():
    return render_template('base.html')
  
@app.route('/about')
def about():
    return render_template('about.html')
      
@app.route('/home')
def home():
    return render_template('home.html')

@crontab.job(minute="0", hour="2", day="*", month="*", day_of_week="*")
@app.route('/retrain')
def retrain2():
    retrain()
    return "Trainiert"
       
@app.route('/suche/<string:search_term>')
def search(search_term: str):
    response = connect_mongo().find(({"$or": [{"title": {"$regex": search_term, "$options": "-i"}},
                                              {"Articledetail": {"$regex": search_term, "$options": "-i"}}
                                              ]}))
    articles = []
    for result in response:
        articles.append(Article.from_mongo_response(result))
    return render_template('suche.html', articles=articles, search_term=search_term.lower())

@app.route('/artikel/<string:post_name>')
def show_article(post_name: str):
    #TOFO: Problem bei a umlaut
    response = connect_mongo().find_one(({"Title": post_name}))
    response_article = Article.from_mongo_response(response)

    return render_template('article.html', article=response_article)

def insert():
    
    params = {
        'spider_name': 'transfer',
        'start_requests': True,
     
    }
    response = requests.get('http://localhost:9080/crawl.json', params)
    data = json.loads(response.text)


    try:
        connect_mongo().insert_many(data['items'])
    except Exception as e:
        print('Fehler!')
    status = {'status': 'ok'}
    
def get_db():
    return db.db
    
def connect_mongo():
    return db.db.collection

def connect_mongo_users():
    return db.db.users

def get_random_id(docids):
    id = randint(0, len(docids))
    return docids[id-1]


@app.route('/scrape', methods=['GET', 'POST'])
def scrape():
    form = startScraping()  # Inst form
    if request.method == 'GET':
        return render_template('scrape.html', form = form)
    insert()
    return render_template('home.html')


@app.route('/scraped', methods=['GET', 'POST'])
def scrapedSites():
    data = get_db().websites.find()
    form = addWebsite()
    if request.method == 'GET':
        return render_template('scrapedsites.html', form=form, data = data)

    new_website = form.correction.data
    
    if new_website is not None:
        websites = data.distinct("websites")
        websites.append(new_website)
        get_db().db.websites.update_one({}, {'$push': {'websites':new_website}})
        print ('Neues Label hinzugefügt')
    return render_template('base.html', choice=new_website, title=new_website, prediction=new_website)


@app.route('/update', methods=['GET', 'POST'])
def update():
    data = get_db().collection.find({"annotated?" : { "$eq" : True}})

    df = pd.DataFrame(list(data))
    return retrain(df)



@app.route('/get', methods=['GET', 'POST'])
def get():    
    next_choice = ''
    next_title = ''
    form = verifyAnswers()  # Inst form

    if request.method == 'GET':
        docids = get_db().collection.find({"annotated?" : { "$ne" : True}, "Articledetail": { "$exists": True }})

        data = None
        id = None
        counter = 0
        for d in docids:
            data = d
            id =  d['_id']
            pred = predict(d['Articledetail'])
            label = pred[0]
            prob =  pred[1]
            counter = counter + 1
            if(max(prob) < 0.51):
                break
        
        if (counter == len(list(docids))):
            id = get_random_id(docids)
            data = get_db().collection.find_one({'_id': ObjectId(data[_id])})    

        pred = predict(data['Articledetail'])
        label = pred[0]
        prob =  pred[1]
        
        #attn_mtrx =  pred[2]
        form.id.data = id
        form.textt.data = data['Title']
        form.text.data = data['Title']
        form.prediction.data = label

        
        title = '<h1>' + data['Title'] + '</h1>'
        subtitle = data['Subtitle']
        articledetail = data['Articledetail']

        try:
            connect_mongo().update_one({'_id': ObjectId(id)}, {'$set': {'lastmodified':datetime.now(), 'ki-prediction': label, 'prob': prob}})
        except Exception as e:
            return('Exception!', e)
   
        return render_template('verify.html',  form=form, title=title, subtitle=subtitle, articledetail=articledetail, id=id, text=title, prob =prob)
    dam = form.answers.data
    id = form.id.data
    usr = form.user.data
    next_choice = dam
    next_title = form.text.data
    next_prediction = form.prediction.data


    try:
        connect_mongo().update_one({'_id': ObjectId(id)}, {'$set': {'user': usr, 'annotated?': True, 'label': dam,'lastmodified':datetime.now()}})
    except Exception as e:
        print(e)
        return('Exception!', e)
    return render_template('base.html', choice=next_choice, title=next_title, prediction=next_prediction)

class verifyAnswers(FlaskForm):
    ''' Form class for verification questions. '''
    labels = get_db().labels.find().distinct('labels')
    dict = {label : label for label in labels}
    
    
    user = SelectField('Current user', choices=[('user1', 'user1'), ('user2', 'user2'), ('user3', 'user3'), ('other', 'other')])
    answers = RadioField('Predicted Dam')

    id = TextField()
    textt = TextAreaField("Description", validators=[InputRequired(),
                                                     Length(max =200)])
    text = TextField()
    prediction = TextField()
    correction = TextField()
    submit = SubmitField()
    
    def __init__(self, *args, **kwargs):
        super(verifyAnswers, self).__init__(*args, **kwargs)

        labels = get_db().labels.find().distinct('labels')
        dict = {label : label for label in labels}
        self.answers.choices = dict.items()
    
    
    
class addWebsite(FlaskForm):
    ''' Form class for add website. '''
    correction = TextField()
    submit = SubmitField()

class startScraping(FlaskForm):
    ''' Form class for start scraping. '''
    submit = SubmitField()


    
if __name__ == '__main__':
    app.config["CACHE_TYPE"] = "null"
    app.run(debug=True, port=1235)
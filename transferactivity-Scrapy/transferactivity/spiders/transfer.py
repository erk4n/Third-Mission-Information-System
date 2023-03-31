import scrapy
import w3lib.html
from datetime import datetime
from flask_pymongo import pymongo
def get_connection():
    CONNECTION_STRING = ""
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('transfer')
    return db

def load_urls(prev_urls):
    prev = []
    for url in prev_urls:
        prev.append(url)
    return prev

class TransferSpider(scrapy.Spider):
    def get_websites():
        CONNECTION_STRING = ""
        client = pymongo.MongoClient(CONNECTION_STRING)
        db = client.get_database('transfer')
        websites = db.websites.find().distinct('websites')
        return websites
    
    for i in get_websites():
        print(i)
    name = 'transfer'
    start_urls = get_websites()
    
    
    def parse(self, response):
        CONNECTION_STRING = ""
        client = pymongo.MongoClient(CONNECTION_STRING)
        db = client.get_database('transfer')
        urls = db.collection.find({}, {'_id':0, 'url': 1 })
    
        print('liste Url',load_urls(urls))
        for articles in response.css('div.outer'):
            for link in articles.css('a.content::attr(href)'):

                if (link.get() in load_urls(urls)):
                    yield response.follow(link.get(), callback = self.parse_articles) #callback: get to that page, follow link and do something
        
         # go to next page
        next_page = response.css("div.next a::attr(href)").extract_first()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
    

        
    def get_url():
        CONNECTION_STRING = ""
        client = pymongo.MongoClient(CONNECTION_STRING)
        db = client.get_database('transfer')
        websites = db.collection.find({}, {'url': 1 });
        return websites

    def parse_articles(self, response):
        
        #labels = get_connection().labels.find().distinct('labels')
        #print(get_connection().collection.find({}, {'url': 1 }))
        articles = response.css('div.outer')
    
        try:
            datumUndAutor = articles.css('div.artikeldetail p::text').get()
            bild = articles.css('div.artikeldetail img').get()
            bildnachweis = articles.css('div.artikeldetail > div.image > p.bildnachweis::text').get()
            print(datumUndAutor.split(',')[0])
            yield {
                'Title': articles.css('span.main::text').get(),
                'Subtitle': articles.css('span.sub::text').get(),
                'Datum': datumUndAutor.split(',')[0],
                'Autor': 'ohne' if datumUndAutor.split(',')[1] is None else datumUndAutor.split(',')[1],
                'Articledetail': w3lib.html.remove_tags(articles.css('div.artikeldetail').get()).replace(datumUndAutor, "").replace(bildnachweis, "").replace("\"",""),
                'Star': 'fest',
                'creationdate': datetime.now(),
                'url': response.request.url,
                'bild': bild,
                'bildnachweis': bildnachweis,
            }
            
        except:
            yield {
                'Title': articles.css('span.main::text').get(),
                'Subtitle': articles.css('span.sub::text').get(),
                'Articledetail': w3lib.html.remove_tags(articles.css('div.artikeldetail').get()).replace(datumUndAutor, "").replace(bildnachweis, "").replace("\"",""),
                'url': response.request.url,
            }


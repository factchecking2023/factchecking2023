import os.path
# from googleapiclient.discovery import build
from readability import Document
import requests
import re
from bs4 import BeautifulSoup
from keys import *
import json
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
# nltk.download('vader_lexicon')
import datetime
import math
import numpy as np
import random
from utils import cred

def extract_date(text):
    pattern = r"\b([A-Z][a-z]{2} \d{1,2}, \d{4})\b"
    match = re.search(pattern, text)
    if match:
        date_str = match.group(1)
        date_obj = datetime.datetime.strptime(date_str, '%b %d, %Y')
        today = datetime.datetime.today()
        delta = today - date_obj
        days_diff = delta.days
        return days_diff
    else:
        # print("No match found.")
        return 0
    

class PageRank:
    def __init__(self) -> None:
        self.PageRankScores = {}
    
    def get_page_rank_score(self, domain):
        if domain.startswith("http"):
            domain = domain.replace("https://", "").replace("http://", "")
            domain = domain.split("/")[0]
        print("domain: ", domain)
        if domain in self.PageRankScores:
            return self.PageRankScores[domain]
    
        headers = {'API-OPR': PAGERANK_API_KEY}
        url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
        print(url)
        request = requests.get(url, headers=headers)
        result = request.json()
        # print(result)
        score = result['response'][0]['page_rank_decimal']
        self.PageRankScores[domain] = score
        return score
    
class GoogleSearch:
    def __init__(self, save_path="google_search_results") -> None:
        self.p = PageRank()
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
    def replace_whitespace(self, text):
        """
        Replace consecutive repeated spaces or line breaks with a single space.
        """
        return re.sub(r'\s+', ' ', text)

    def get_main_text(self, url):
        # url = "https://headtopics.com/us/did-marines-catch-and-kill-fbi-agents-trying-to-sabotage-a-substation-in-idaho-35666206"
        try:
            to_file = "{}/{}.html".format(self.save_path, re.sub('[\W_]+', '', url)[:80])
            if not os.path.exists(to_file):
                print(url)
                response = requests.get(url)
                response.encoding = "utf-8"
                with open(to_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("saved to", to_file)
            with open(to_file, encoding="utf-8") as f:
                html = f.read()
            
            doc = Document(html)
            title = doc.title()
            main_text = doc.summary()

            # print(title)
            soup = BeautifulSoup(main_text, 'html.parser')
            return title, self.replace_whitespace(soup.getText())
        except Exception as e:
            print("Error:", str(e))

        return '', ''

    def google_search_api(self, search_term):
        # service = build("customsearch", "v1", developerKey=api_key)
        # res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        # return res
        url = f"https://www.googleapis.com/customsearch/v1"
        data = requests.get(url, params={
            'key': GOOGLE_API_KEY2,
            'cx': GOOGLE_CSE_ID2,
            'q': search_term
        }).json()
        return data


    def google_search(self, claim):
        print("Claim: ", claim)
        filename = re.sub('[\W_]+', '', claim)[:80]

        to_file = "{}/{}.json".format(self.save_path, filename)
        if not os.path.exists(to_file):
            result = self.google_search_api(claim)
            json_object = json.dumps(result, indent = 4)
            with open(to_file, "w") as f:
                f.write(str(json_object))
            print("saved to", to_file)
        else:
            print(to_file)

        with open(to_file) as f:
            search_result = json.load(f)
        return search_result

    def search_for_reference(self, claim):
        claim_file = os.path.join(self.save_path, "claim.txt")
        with open(claim_file, "w") as f:
            f.write(claim.strip() + "\n")
        print("saved to", claim_file)

        search_result = self.google_search(claim)
        sia = SentimentIntensityAnalyzer()

        with open("skippedsites.txt") as f:
            skippedsites = [l.strip() for l in f.read().strip().split("\n")]
        # print("socialmedias:", socialmedias)

        links = []

        i = 0
        w1 = 0.5
        w2 = 0.5
        for item in search_result['items']:
            print("Item:", i)
            print("Title:", item['title'])
            print(item['snippet'])
            print("Link:", item['link'])
        

            domain = item['displayLink']
            flag = False
            for s in skippedsites:
                if s.find(domain) > -1:
                    flag = True

            if flag:
                i += 1
                continue
            
            
            sentiment_score = sia.polarity_scores(item['snippet'])["compound"]
            page_rank_score = self.p.get_page_rank_score(item['displayLink'])
            days = extract_date(item['snippet'])
            cred_score = cred(days, i, page_rank_score)

            main_text = self.get_main_text(item['link'])
            links.append( {
                'item_index': i,
                'title': item['title'],
                'link': item['link'],
                'snippet': item['snippet'],
                'domain': item['displayLink'],
                'page_rank': page_rank_score,
                'sentiment_score': sentiment_score,
                'days': days,
                'credibility_score': cred_score,
                'main_text': main_text
            })
            print("Domain:", item['displayLink'])
            print("PageRank score:", page_rank_score)
            print("Sentiment score:", sentiment_score)
            print("days:",  days )
            print("Credibility score:", cred_score)

            ref_file = os.path.join(self.save_path, "ref_{}.txt".format(i))
            with open(ref_file, "w") as f:
                f.write("URL: {} \n".format(item['link'].strip()))
                f.write("Title:\n")
                f.write("Body:\n")
            print("saved to {}".format(ref_file))
            print()

            i += 1
        
        links.sort(key=lambda x: x['credibility_score'], reverse=True)
        
        res = {
            'claim': claim, 
            'references': links
        }

        cache_file = os.path.join(self.save_path, "google_search_results.json")
        with open(cache_file, "w") as f:
            f.write( json.dumps(res, ensure_ascii=False, indent=4) )
        print("run the following command to cp file to another server:")
        print("    scp {} {}".format(cache_file, MY_SERVER_DATAPATH))
        return res


if __name__ == '__main__':
    claim = "U.S. Marines \"gunned down criminal FBI agents trying to sabotage an electrical substation in Meridian, Idaho.\""
    print(claim)
    g = GoogleSearch()
    
    res = g.search_for_reference(claim)
    print(json.dumps(res, indent=4))
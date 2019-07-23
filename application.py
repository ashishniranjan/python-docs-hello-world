

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 23:00:31 2019

@author: niran
"""

from flask import Flask,abort,jsonify,request
import nltk
import numpy as np
import random
import string

raw = "India (ISO: Bhārat), official name: the Republic of India,[e] (ISO: Bhārat Gaṇarājya), is a country in South Asia. It is the seventh-largest country by area, the second-most populous country, and the most populous democracy in the world. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[f] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia. The Indian subcontinent was home to the Indus Valley Civilisation of the bronze age. In India's iron age, the oldest scriptures of Hinduism were composed, social stratification based on caste emerged, and Buddhism and Jainism arose. Political consolidations took place under the Maurya and Gupta Empires; the peninsular Middle Kingdoms influenced the cultures of Southeast Asia. In India's medieval era, Judaism, Zoroastrianism, Christianity, and Islam arrived, and Sikhism emerged, adding to a diverse culture. North India fell to the Delhi Sultanate; south India was united under the Vijayanagara Empire. In the early modern era, the expansive Mughal Empire was followed by East India Company rule. India's modern age was marked by British Crown rule and a nationalist movement which, under Mahatma Gandhi, was noted for nonviolence and led to India's independence in 1947."
raw = raw.lower()
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def response():
    data = request.get_json(force=True)
    user_response = data['body']
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return jsonify(results=robo_response)
    else:
        robo_response = robo_response+sent_tokens[idx]
        return jsonify(results=robo_response)
    
if __name__ == '__main__':
    app.run(port=9000,debug=True)

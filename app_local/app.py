from pyngrok import ngrok
from flask import Flask, render_template , request, send_file
import os
import threading
import pickle
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
from pywsd.similarity import max_similarity
import requests
import json
from wn import WordNet
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn

nltk.download('stopwords')
nltk.download('popular')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from answerquest import QnAPipeline
import nlp2go
import subprocess
#subprocess.call("answerquest/download_models.sh",shell=True)
class InputForm(FlaskForm):
    inputText = TextAreaField('Enter Quiz Material', validators=[DataRequired()],render_kw={'rows':'10'})
    submit = SubmitField('Generate Questions')

def get_wordsense(sent,word):
    word= word.lower()
    
    if len(word.split())>0:
        word = word.replace(" ","_")
    
    
    synsets = wn.synsets(word,'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
        lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_distractors_conceptnet(word):
    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = [] 
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term'] 

        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)
                   
    return distractor_list

def initialise(qg_tokenizer_path, qg_model_path, qa_model_path ):
  qna_pipeline = QnAPipeline(
    qg_tokenizer_path=qg_tokenizer_path,
    qg_model_path=qg_model_path,
    qa_model_path=qa_model_path
  )
  return qna_pipeline
qna_pipeline = initialise("qg_augmented_model/gpt2_tokenizer_vocab/", "qg_augmented_model/qg_augmented_model.pt", "doc_qa_model/checkpoint-59499")

# def func(input_text):
#     qna_pipeline, bdg_pm_model = initialise("/content/gdrive/MyDrive/IR_code/qg_augmented_model/gpt2_tokenizer_vocab/", "/content/gdrive/MyDrive/IR_code/qg_augmented_model/qg_augmented_model.pt", "/content/gdrive/MyDrive/IR_code/doc_qa_model/checkpoint-59499", '/content/gdrive/MyDrive/IR_code/BDG_PM.pt')

#     (sent_idxs,questions,answers) = qna_pipeline.generate_qna_items(
#         input_text,
#         filter_duplicate_answers=True,
#         filter_redundant=True,
#         sort_by_sent_order=True)

#     # sent_idxs, questions, answers
#     distractors = []

#     result = []

#     for a,b,c in zip(sent_idxs, questions, answers):
#         print(a)

#         strI = input_text + " [SEP] " + b + " [SEP] " + c
#         input_json = {
#         "input": strI
#         }

#         x=bdg_pm_model.predict(input_json)
#         new_entry = dict()
#         new_entry['question']=b
#         new_entry['line_no']=a
#         new_entry['ans']=c
#         new_entry['distractor']=x['result'][0]
#         result.append(new_entry)

#     return result
def func(qna_pipeline, input_text):
  #qna_pipeline = initialise("/content/gdrive/MyDrive/IR_code/qg_augmented_model/gpt2_tokenizer_vocab/", "/content/gdrive/MyDrive/IR_code/qg_augmented_model/qg_augmented_model.pt", "/content/gdrive/MyDrive/IR_code/doc_qa_model/checkpoint-59499")
  (sent_idxs,questions,answers) = qna_pipeline.generate_qna_items(
    input_text,
    filter_duplicate_answers=True,
    filter_redundant=True,
    sort_by_sent_order=True)
  
  keyword_sentence_mapping = dict()

  for i,j in zip(answers, questions):
    keyword_sentence_mapping[i]=[j]


  key_distractor_list = {}
  # keyword_sentence_mapping = d

  for keyword in keyword_sentence_mapping:
    wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
    if wordsense:
      distractors = get_distractors_wordnet(wordsense,keyword)
      if len(distractors) ==0:
        distractors = get_distractors_conceptnet(keyword)
      if len(distractors) != 0:
        key_distractor_list[keyword] = [keyword_sentence_mapping[keyword][0]]+distractors
    else:   
      distractors = get_distractors_conceptnet(keyword)
      if len(distractors) != 0:
        key_distractor_list[keyword] = [keyword_sentence_mapping[keyword][0]]+distractors

  QADlist = []

  for ans, value in key_distractor_list.items():
    d = dict()
    d['ans'] = ans
    d['question'] = value[0]
    d['distractor'] = value[1:]

    QADlist.append(d)
  
  return QADlist 


app = Flask(__name__)#,template_folder='/templates',static_folder='/static')
#ngrok.set_auth_token("27eM94qtit5orPFHdi2ZynwCk45_24A5V5MWUSx5o2ECzKiuh")
#port = 5000
#public_url = ngrok.connect(port).public_url
# from google.colab.output import eval_js
# print(eval_js("google.colab.kernel.proxyPort(5000)"))
#app.config["BASE_URL"] = public_url
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'
Bootstrap(app)

# qna_pipeline= None
# bdg_pm_model=None

# @app.before_first_request
# def activate_job():
#   qna_pipeline, bdg_pm_model = initialise("/content/gdrive/MyDrive/IR_code/qg_augmented_model/gpt2_tokenizer_vocab/", "/content/gdrive/MyDrive/IR_code/qg_augmented_model/qg_augmented_model.pt", "/content/gdrive/MyDrive/IR_code/doc_qa_model/checkpoint-59499", '/content/gdrive/MyDrive/IR_code/BDG_PM.pt')

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/quizgen',methods=["GET", "POST"])
# def quizgen():
    
#     form = InputForm()
#     qa = []
#     if form.validate_on_submit():
#         inputData = form.inputText.data
#         qa = func(qna_pipeline, inputData)
#         print(qa)
#         #return render_template('quizgen.html',form=form,result=qa)
#     return render_template('quizgen.html',form=form,results=qa)
#print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

@app.route('/quizgen',methods=["GET", "POST"])
def quizgen():
    
    form = InputForm()
    qa = []
    if form.validate_on_submit():
        inputData = form.inputText.data
        qa = func(qna_pipeline, inputData)
        print(qa)
        with open('filename.txt', 'w') as convert_file:
          convert_file.write(json.dumps(qa))
        #return render_template('quizgen.html',form=form,result=qa)
    return render_template('quizgen.html',form=form,results=qa)

@app.route('/download')
def download():
	path = "filename.txt"
	return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run()



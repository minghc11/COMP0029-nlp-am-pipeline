import csv
import os
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def reader():
    claim_df = pd.read_csv('../ibm_claim_evidence_dataset/claims_2015.txt', sep='\t')
    claims = claim_df['Claim corrected version'].tolist()
    return claims

folder_path = '../ibm_claim_evidence_dataset/articles'
articles = {}

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        articles[filename] = file.read()

df = pd.read_csv('../ibm_claim_evidence_dataset/articles.txt', sep='\t', usecols=['article Id', 'Topic'])

articles_info = df.set_index('article Id').to_dict()['Topic']

def is_claim(sentence, claims):
    return any(claim in sentence for claim in claims)

claims = reader()

labelled_sentences = []
for article_id, topic in articles_info.items():
    filename = f'clean_{article_id}.txt'
    content = articles[filename]
    sentences = sent_tokenize(content)
    for sentence in sentences:
        label = 1 if is_claim(sentence, claims) else 0
        labelled_sentences.append({"sentence": sentence, "label": label, "topic": topic})

with open('../processed_dataset/claims.csv', "w", encoding='utf-8', newline='') as f:
    fieldnames = ['sentence', 'label', 'topic']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(labelled_sentences)



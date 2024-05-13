import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import os
from collections import defaultdict

df = pd.DataFrame(columns=['Claim', 'Candidate', 'Label'])

evidence_df = pd.read_csv('../ibm_claim_evidence_dataset/evidence_2015.txt', sep='\t')

folder_path = '../ibm_claim_evidence_dataset/articles'
articles = {}

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        articles[filename] = file.read()

def create_candidates(sentences):
    candidates = []
    for i in range(len(sentences)):
        candidates.append((sentences[i],))  # One sentence
        if i < len(sentences) - 1:
            candidates.append((sentences[i], sentences[i + 1]))  # Two sentences
        if i < len(sentences) - 2:
            candidates.append((sentences[i], sentences[i + 1], sentences[i + 2]))  # Three sentences
    return candidates

def label_evidence(candidate, evidences):
    full_text = ' '.join(candidate)
    for evidence in evidences:
        evidence_lower = evidence.lower()
        full_text_lower = full_text.lower()
        if evidence_lower in full_text_lower:
            if len(candidate) > 1 and any(evidence_lower in s.lower() for s in candidate):
                return 0 # label multi-sentence candidate with evidence within a single sentence as 0
            
            if len(candidate) == 3:
                if not any(evidence_lower in s.lower() for s in candidate):
                    return 0 # label three-sentence candidates with evidence spanning over two sentences as 0
            
            return 1
    return 0

claim_evidence_dict = defaultdict(list)
for _, row in evidence_df.iterrows():
    claim = row['Claim']
    evidence = row['Evidence']

    claim_evidence_dict[claim].append(evidence)

claim_evidence_dict = dict(claim_evidence_dict)
print(len(claim_evidence_dict))

article_count = 0
for title, content in articles.items():
    article_count += 1
    print("article", article_count)
    
    sentences = sent_tokenize(content)
    candidates = create_candidates(sentences)

    count = 0
    for claim, evidences in claim_evidence_dict.items():
        if claim.lower() not in content:
            continue

        count += 1
        if count % 100 == 0:
            print(count)
        
        for candidate in candidates:
            full_text = ' '.join(candidate)
            label = label_evidence(candidate, evidences)
            temp_data = {'Claim': claim,
                                'Candidate': full_text,
                                'Label': label}
            
            temp_df = pd.DataFrame([temp_data])
            
            df = pd.concat([df, temp_df], ignore_index=True)

df.drop_duplicates()
output_path = '../processed_dataset/evidence.csv'
df.to_csv(output_path, index=False)
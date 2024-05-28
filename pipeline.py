import torch
from BERTClassifier import BERTClassifier
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
claim_bert_model_name = 'bert-large-cased' 
claim_tokenizer = BertTokenizer.from_pretrained(claim_bert_model_name)

evidence_bert_model_name = 'bert-base-cased'
evidence_tokenizer = BertTokenizer.from_pretrained(evidence_bert_model_name)

stance_bert_model_name = 'bert-large-cased'
stance_tokenizer = BertTokenizer.from_pretrained(stance_bert_model_name)

relation_bert_model_name = 'bert-large-cased'
relation_tokenizer = BertTokenizer.from_pretrained(relation_bert_model_name)

def evaluateModel(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

class InputDataset(Dataset):
    def __init__(self, argument1s, argument2s, tokenizer, max_length):
        self.argument1s = argument1s
        self.argument2s = argument2s
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.argument1s)
    def __getitem__(self, idx):
        argument1 = self.argument1s[idx]
        argument2 = self.argument2s[idx]
        encoding = self.tokenizer(
            argument1,
            argument2,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}
    
def create_candidates(sentences):
    candidates = []
    for i in range(len(sentences)):
        candidates.append(sentences[i])  # One sentence
        if i < len(sentences) - 1:
            candidates.append(sentences[i] + ' ' + sentences[i + 1])  # Two sentences
        if i < len(sentences) - 2:
            candidates.append(sentences[i] + ' ' + sentences[i + 1] + ' ' + sentences[i + 2])  # Three sentences
    return candidates

def get_predictions(dataloader, model):
    predictions = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().tolist())

    return predictions

def get_claims(sentences, topic):
    topics = [topic] * len(sentences)

    claim_dataset = InputDataset(sentences, topics, claim_tokenizer, 256)
    claim_dataloader = DataLoader(claim_dataset, batch_size=16)

    model_claim = BERTClassifier(claim_bert_model_name, 2).to(device)
    model_claim.load_state_dict(torch.load("./final_models/claim.pth", map_location=device))
    model_claim.eval()

    claim_predictions = get_predictions(claim_dataloader, model_claim)

    claims = []
    for sentence, claim_prediction in zip(sentences, claim_predictions):
        if claim_prediction == 1:
            claims.append(sentence)

    return claims

def get_evidences(sentences, result):
    for claim, _ in result.items():
        evidence_candidates = create_candidates(sentences)
        multiple_claim = [claim] * len(evidence_candidates)

        evidence_dataset = InputDataset(multiple_claim, evidence_candidates, evidence_tokenizer, 256)
        evidence_dataloader = DataLoader(evidence_dataset, batch_size=16)

        model_evidence = BERTClassifier(evidence_bert_model_name, 2).to(device)
        model_evidence.load_state_dict(torch.load("./final_models/evidence.pth", map_location=device))
        model_evidence.eval()

        evidence_predictions = get_predictions(evidence_dataloader, model_evidence)
        evidences = []
        for candidate, evidence_prediction in zip(evidence_candidates, evidence_predictions):
            if evidence_prediction == 1:
                evidences.append(candidate)
        
        result[claim]['evidence'] = evidences
        
    return result

def get_stances(claims, topic, result):
    topics = [topic] * len(claims)
    stance_dataset = InputDataset(claims, topics, stance_tokenizer, 256)
    stance_dataloader = DataLoader(stance_dataset, batch_size=16)

    model_stance = BERTClassifier(stance_bert_model_name, 2).to(device)
    model_stance.load_state_dict(torch.load("./final_models/stance.pth", map_location=device))
    model_stance.eval()

    stance_predictions = get_predictions(stance_dataloader, model_stance)
    
    for claim, stance in zip(claims, stance_predictions):
        if stance == 0:
            result[claim]['stance'] = 'CON'
        elif stance == 1:
            result[claim]['stance'] = 'PRO'

    return result   

def get_relations(claims):
    argument1 = []
    argument2 = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            argument1.append(claims[i])
            argument2.append(claims[j])

    relation_dataset = InputDataset(argument1, argument2, relation_tokenizer, 512)
    relation_dataloader = DataLoader(relation_dataset, batch_size=16)

    model_relation = BERTClassifier(relation_bert_model_name, 3).to(device)
    model_relation.load_state_dict(torch.load("./final_models/relation.pth", map_location=device))
    model_relation.eval()

    relation_predictions = get_predictions(relation_dataloader, model_relation)

    relations = {
        'support': [],
        'attack': [],
        'no_relation': []
    }

    for a1, a2, relation_prediction in zip(argument1, argument2, relation_predictions):
        if relation_prediction == 0:
            relations['attack'].append((a1, a2))
        elif relation_prediction == 1:
            relations['support'].append((a1, a2))
        elif relation_prediction == 2:
            relations['no_relation'].append((a1, a2))

    return relations

def pipeline(text, topic):
    sentences = sent_tokenize(text)

    claims = get_claims(sentences, topic)
    result = {}
    for claim in claims:
        result[claim] = {}

    result = get_evidences(sentences, result)
    result = get_stances(claims, topic, result)

    relations = get_relations(claims)

    result['relations'] = relations

    return result

topic = input('Please enter a topic: ')
article = input('Please enter some texts: ')
result = pipeline(article, topic)
print(result)
print(len(result))
json_file_path = './output/output.json'
with open(json_file_path, "w") as json_file:
    json.dump(result, json_file, indent=4)
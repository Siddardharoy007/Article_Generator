import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("India won the 2023 Cricket World Cup under Rohit Sharma's captaincy.")

for ent in doc.ents:
    print(ent.text, ent.label_)

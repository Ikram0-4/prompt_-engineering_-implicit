import pandas as pd 
import xml.etree.ElementTree as xml
import torch
import numpy
import os
from openai import OpenAI
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import matplotlib
from huggingface_hub import login
"Parser le xml et extraire les balise s pour phrase et leur attribut sentence pour récupérer la phrase, choisir 100 dont l'attribut ana est à valeur 0 et pareil avec valeur à 1. Le premier indique un déclencheur EFECTIF de présupoistion et le second la négative."
file_annoted_byIkram = "triggers_cassation-corpus.xml"
tree = xml.parse(file_annoted_byIkram)
root = tree.getroot()

data_ana0 = []
data_ana1 = []

for s in root.findall(".//s"):
    sentence_text = s.get("sentence", "").strip()

    # Chercher tous les triggers dans ce <s>
    triggers = s.findall("trigger")
    anas_in_s = [t.get("ana") for t in triggers if t.get("ana") in ("0", "1")]

    # On regarde si au moins un trigger a ana=0 ou ana=1
    if "0" in anas_in_s and len(data_ana0) < 100:
        data_ana0.append({"ana": "0", "sentence": sentence_text})
    if "1" in anas_in_s and len(data_ana1) < 100:
        data_ana1.append({"ana": "1", "sentence": sentence_text})

    if len(data_ana0) >= 100 and len(data_ana1) >= 100:
        break

data = data_ana0 + data_ana1
df = pd.DataFrame(data)
df.to_csv("output_ana.csv", index=False, sep=";")

print(f"CSV créé avec {len(data_ana0)} phrases ana=0 et {len(data_ana1)} phrases ana=1")

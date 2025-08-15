import torch
from openai import OpenAI
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import matplotlib
from huggingface_hub import login
import pandas as pd
import os

token_mistral = os.getenv("MISTRAL_KEY")
token_openia_gpt5 = os.getenv("OPEN_AI_KEY")


df = pd.read_csv("output_ana.csv", sep=";")

print(df.head())


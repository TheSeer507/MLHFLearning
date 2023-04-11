from transformers import pipeline
import torch
import torch.nn.functional as F

classifier = pipeline("sentiment-analysis")
results = classifier("This is my first test using HuggingFace, so happy for this milestone.",
                     "I hope I come to enjoy it rather than hate it")

for result in results:
    print(result)
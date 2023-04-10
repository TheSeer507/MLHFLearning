import pandas as pd
import numpy as np
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

res = classifier("This is my first test using HuggingFace, so happy for this milestone.")

print(res)
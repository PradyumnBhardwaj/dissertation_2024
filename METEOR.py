from nltk.translate.meteor_score import meteor_score
import nltk
from nltk.tokenize import word_tokenize
import ollama

nltk.download('punkt')
nltk.download('wordnet')
response = ollama.chat(model='falcon:7b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
generated_response = response['message']['content']
tokenized_generated_response = word_tokenize(generated_response)
reference_response = ["my name is rishBH"]
tokenized_reference_response = [word_tokenize(ref) for ref in reference_response]
score = meteor_score(tokenized_reference_response, tokenized_generated_response)

print(f'METEOR Score: {score}')

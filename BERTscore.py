import ollama
from evaluate import load
bertscore = load("bertscore")
response = ollama.chat(model='falcon:7b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

generated_response = response['message']['content']
print("Generated Response:", generated_response)
ans=[generated_response]
references = ["hello there"]
results = bertscore.compute(predictions=ans, references=references, lang="en")
print(f"Precision: {results['precision'][0]}")
print(f"Recall: {results['recall'][0]}")
print(f"F1 Score: {results['f1'][0]}")
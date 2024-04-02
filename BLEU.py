from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import ollama
source_texts = [
    "Why is the sky blue?",
]
reference_translations = [
    ["The sky is blue due to the scattering of sunlight by the atmosphere. The blue color of the sky is the result of a particular type of scattering called Rayleigh scattering."],
]

def get_model_translations(model, messages):
    responses = []
    for message in messages:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': message}])
        responses.append(response['message']['content'])
    return responses

candidates = get_model_translations('falcon:7b', source_texts)
print(candidates)

references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]
chencherry = SmoothingFunction()
score = corpus_bleu(references_tokenized, candidates_tokenized, smoothing_function=chencherry.method1)

print(f"BLEU score: {score}")
  
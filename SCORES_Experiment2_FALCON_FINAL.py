from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from rouge import Rouge
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import ollama
from evaluate import load

# Download necessary NLTK models if not already done
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Function to get model translations using Ollama
def get_model_translations(model, messages):
    responses = []
    for message in messages:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': message}])
        responses.append(response['message']['content'])
    return responses

# Calculate the Translation Edit Rate (TER) score
def calculate_ter(generated_texts, reference_texts):
    ter_scores = []
    for candidate, refs in zip(generated_texts, reference_texts):
        references = [[ref] for ref in refs]  
        ter = sacrebleu.corpus_ter([candidate], references)  
        ter_scores.append(ter.score)
    return ter_scores

# Load BERTScore from evaluate library
bertscore = load("bertscore")

reference_translations = [
    [" This Earth Day, let's make every charge count for the planet! Introducing our Solar-Powered Chargers — the perfect companion for the eco-conscious tech enthusiast. Harness the power of the sun to keep your devices charged and your ecological footprint minimal. Ready to power your adventures sustainably?"],
    ["In an era where technology and environmental sustainability seem at odds, a new wave of innovation is proving otherwise. Sustainable gadgets are not just a trend; they're revolutionizing the tech industry, shifting paradigms towards a greener future. In this article, we explore five key ways these eco-friendly innovations are making their mark, from reducing waste to harnessing renewable energy. Join us as we delve into how sustainable gadgets are setting the new standard for the tech world."],
    ["Capture the world in stunning clarity while keeping the planet green with our EcoSmart Camera. Engineered with the environment in mind, this camera combines cutting-edge technology with eco-friendly features. Its durable, recycled casing reduces waste, while the low-energy consumption design ensures that your photographic adventures are both breathtaking and sustainable. Perfect for the eco-conscious photographer who does not want to compromise on quality or the planet’s health. With the EcoSmart Camera, every shot contributes to a greener tomorrow."],
    ["Dear EcoWarriors, This month, we're thrilled to bring you the latest in green technology innovations and exclusive offers from EcoFriendly Gadgets. As the world moves towards a more sustainable future, we're at the forefront, ensuring that our products not only elevate your lifestyle but also protect the planet. From groundbreaking eco-smart devices to our commitment to reducing electronic waste, we're excited to share how our innovations are contributing to a greener world. Plus, don't miss out on our special Earth Month discounts, available for a limited time. Join us in making sustainability the standard."]
]

# Get model translations
# candidates = get_model_translations('falcon:7b', source_texts)

candidates = ["Generate an engaging social media post for Earth Day promoting our solar-powered chargers. Create a social media post that promotes our Solar Chargers and their sustainability benefits to help drive awareness around our product and Earth Day.",
              "$30-250 USD We're looking for someone to create an engaging introduction for the article below. We're open to the style of writing - it could be an engaging blog post, a persuasive article, anything that is engaging and gets the message across. The article will be 5 - 10 000 words in length and will be published on our website. This topic is a little different to the usual articles we post and it's a bit of a niche subject but we think it can be a really interesting and engaging piece.",
              "This contest ends at 9am on Thursday 5th July 2018, so be quick! The best entries will be shared with you. The prize will be a £100 Amazon gift voucher. The winning description will appear in the EcoSmart product description in our online store and will be used in our marketing materials. Please ensure your description is original, and that you have obtained all relevant permissions/clearances. Your description must be written in English. Please use the form below to submit your entry. Please ensure the description is 1000 characters or less.",
              "Hi @Lisamarie_L, I apologize for any confusion you're experiencing. As I understand you're having issues with the chat feature not connecting with users on the website. I've done some testing on your website and was unable to replicate the chat not connecting. I am able to connect and chat from the website. I'm sorry for any inconvenience you're experiencing."
              ]

# Display Generated Text
for candidate in candidates:
    print("Generated Text:", candidate)

# Tokenize sentences for BLEU calculation
references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]

chencherry = SmoothingFunction()

print("FALCON- Experiment-2")

for i, candidate_tokenized in enumerate(candidates_tokenized):
    score = corpus_bleu([references_tokenized[i]], [candidate_tokenized], smoothing_function=chencherry.method1)
    print(f"BLEU score for candidate {i+1}: {score}")

# Initialize the Rouge scorer and calculate scores
rouge = Rouge()
rouge_scores = rouge.get_scores(candidates, [refs[0] for refs in reference_translations], avg=False)

# Extracting and printing ROUGE scores
for i, scores in enumerate(rouge_scores):
    print(f"Candidate {i+1} ROUGE-1: r: {scores['rouge-1']['r']:.4f}, p: {scores['rouge-1']['p']:.4f}, f: {scores['rouge-1']['f']:.4f}")
    print(f"Candidate {i+1} ROUGE-2: r: {scores['rouge-2']['r']:.4f}, p: {scores['rouge-2']['p']:.4f}, f: {scores['rouge-2']['f']:.4f}")
    print(f"Candidate {i+1} ROUGE-L: r: {scores['rouge-l']['r']:.4f}, p: {scores['rouge-l']['p']:.4f}, f: {scores['rouge-l']['f']:.4f}")

# Calculate TER score
ter_scores = calculate_ter(candidates, reference_translations)

for i, score in enumerate(ter_scores):
    print(f"Candidate {i+1} TER Score: {score:.2f}")

# Tokenize for METEOR score calculation
tokenized_candidates = [word_tokenize(cand) for cand in candidates]
tokenized_reference_translations = [[word_tokenize(ref) for ref in refs] for refs in reference_translations]

# Calculate METEOR scores
meteor_scores = [meteor_score(refs, cand) for cand, refs in zip(tokenized_candidates, tokenized_reference_translations)]

for i, score in enumerate(meteor_scores):
    print(f"METEOR Score for text {i+1}: {score:.4f}")
# Compute BERT scores
results = bertscore.compute(predictions=candidates, references=reference_translations, lang="en")

# Print Precision, Recall, and F1 Score for each prediction-reference pair
for i in range(len(candidates)):
    print(f"BERTScore Text {i+1} - Precision: {results['precision'][i]}, Recall: {results['recall'][i]}, F1 Score: {results['f1'][i]}")



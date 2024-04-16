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
    ["Recent market research highlights a surge in the renewable energy sector, with solar panels experiencing increased demand due to their cost-effectiveness and sustainability. Wind energy investments are also on the rise, indicating a robust interest in alternative energy sources. Moreover, Southeast Asia is emerging as a significant market for renewable energy, driven by growing environmental awareness and the region's potential for solar and wind energy production."],
    ["The tech industry is witnessing significant growth, particularly in AI and machine learning applications, transforming everything from consumer products to business processes. Concurrently, cybersecurity threats have become more prevalent, emphasizing the need for advanced security solutions. Additionally, the shift towards remote work has accelerated the adoption of cloud computing services, facilitating access to resources and collaboration from anywhere."],
    ["The e-commerce sector has experienced unprecedented growth, largely accelerated by global health events that have pushed consumers towards online shopping. This shift has been supported by a rise in digital payment platforms, which offer convenient and secure transaction options. To accommodate the increased volume of online sales, logistics companies have notably expanded their capabilities, enhancing delivery services to meet consumer demand efficiently."],
    ["There's a growing consumer demand for sustainable and ethically produced fashion, indicating a shift towards more environmentally conscious purchasing decisions. Brands are increasingly committing to sustainable practices, including the use of eco-friendly materials and more ethical production processes. This trend reflects a broader move towards sustainability in the fashion industry, aiming to reduce environmental impact and improve social responsibility."],
    ["The widespread shift to remote work has significant implications for urban development, leading to a decreased demand for large office spaces in urban centers. In contrast, there's an increasing demand for residential properties that can accommodate home offices, suggesting a change in living preferences towards more versatile living spaces. This shift is prompting urban planners and developers to rethink the design and functionality of residential areas to meet the evolving needs of a remote workforce."]
]


# Get model translations
# candidates = get_model_translations('falcon:7b', source_texts)

candidates = ["The global demand for energy is growing, but so too is the demand for renewable energy. Renewable energy, which includes solar, wind, hydroelectric (or hydro) and other forms of sustainable energy, is a rapidly evolving sector, which is experiencing a significant increase in demand for equipment, services and technologies across Asia, Europe and North America. The market is forecast to double from $160bn in 2016 to $330bn in 2024, according to a recent report by GlobalData.",
              "The post Latest Industry Insights from the 2019 IT Salary Survey appeared first on The Official Cloudwards.net Blog. via Cloudwards.net The post Latest Industry Insights from the 2019 IT Salary Survey appeared first on AroundTheBit.",
              "In this webinar, we look at the logistics industry's response to the global pandemic, its challenges and the new business models that have emerged, as well as the opportunities ahead for the global logistics industry. Watch this webinar now! Speakers: David Hough Head of Global Logistics & Supply Chain, SES Networks David Hough is responsible for leading SES’ Global Satellite Logistics Business. ",
              "This is not a passing trend. According to a survey conducted in 2017 by the National Retail Federation, 75% of US consumers said they are more likely to make a purchase from a socially conscious business. The fashion industry’s role in environmental degradation is well-documented. The apparel market accounts for 10% of all global carbon emissions. That’s more than all international air travel and cruises combined. ",
              "The shift to remote work is influencing urban development, with a decreasing need for large office spaces and an increasing demand for residential areas with home office capabilities. The trend towards working from home has increased the demand for residential spaces that can offer home office capabilities. As a result, the demand for residential property located in urban centers with excellent public transport links and a high density of office space has increased significantly. The shift to remote work is influencing urban development, with a decreasing need for large office spaces and an increasing demand for residential areas with home office capabilities. The shift to remote working is influencing the future of work by influencing the need for office space. "
              ]

# Display Generated Text
for candidate in candidates:
    print("Generated Text:", candidate)

# Tokenize sentences for BLEU calculation
references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]

chencherry = SmoothingFunction()

print("FALCON- Experiment-4")

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



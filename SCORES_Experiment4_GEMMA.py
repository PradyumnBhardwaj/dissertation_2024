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
        references = [[ref] for ref in refs]  # Ensuring the reference texts are in the correct format
        ter = sacrebleu.corpus_ter([candidate], references)  # Calculate TER for each candidate against its references
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

candidates = ["The renewable sector is growing rapidly, with a 10% increase in global installed capacity in 2021. This growth is driven by increased investment, technological advancements, government policies, consumer demand, environmental concerns, cost-effectiveness, energy security, job creation, economic growth, innovation, sustainability, diversification, reliability, resilience, competitiveness, efficiency, scalability, flexibility, adaptability, integration, interoperability, compatibility, standardization, harmonization, certification, regulation, governance, transparency, accountability, collaboration, partnerships, networks, alliances, consortia, clusters, ecosystems, platforms, marketplaces, exchanges, markets, trading, hedging, risk management, asset management.",
              "The tech sector is constantly evolving, with new trends emerging and existing ones gaining momentum. In this article, we’ll take a look at some of the most significant tech trends that are shaping the industry in 2023. From the growing use of artificial intelligence and cloud-based services to the increasing importance of cybersecurity and data privacy, these trends are set to have a major impact on how businesses operate and how consumers interact with technology. Whether you’re a tech enthusiast or a business owner looking to stay ahead of industry developments, this overview of tech’s latest trends is a must-read. So, let’ s dive in and explore the future of technology! <h2>Artificial Intelligence and Machine Learning</h2> Artificial intelligence (AI) and its subset, machine-learning, are two of today’ most talked",
              "The e-Commerce industry is expected to grow by 20% in <strong>2</strong><strong>023</strong>, reaching a market size of 1.3 trillion dollars. This growth is driven by the increasing popularity of online marketplaces, the rise of social commerce, advancements in technology, such as artificial intelligence and machine learning, as well as the growing demand for personalized and convenient shopping experiences. Additionally, eWallets and digital payments are becoming more popular, making it easier for consumers to make purchases online. The eCommerce market is also expected to be driven in part by an increase of mobile commerce. As more people use their smartphones to shop online, businesses will need to adapt their websites and apps to provide a seamless and user-friendly experience.",
              "But what about the end of the product’s life cycle? The fashion industry is one of many that are guilty of producing more than it can consume. In fact, the average American throws away 81 pounds of clothing each year. That’d be the equivalent of 100 pairs of jeans. And the problem is only getting worse. The average consumer buys 60% more clothing than they did 20 years ago. But they keep each item for half as long.  So, what can we do to reduce the amount of waste we produce? One solution is to recycle our clothes. This is where the concept of a circular economy comes in. A circular fashion economy is a system that aims to eliminate waste and pollution, circulate products and materials, and regenerate nature. It’ll help us to ",
              "The pandemic has also accelerated the trend of people moving to the suburbs, where they can enjoy more space and a better quality of life. This is leading to a shift in the way cities are designed, as developers focus on creating more livable and sustainable communities. "
              ]

# Display Generated Text
for candidate in candidates:
    print("Generated Text:", candidate)

# Tokenize sentences for BLEU calculation
references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]

chencherry = SmoothingFunction()


for i, candidate_tokenized in enumerate(candidates_tokenized):
    score = corpus_bleu([references_tokenized[i]], [candidate_tokenized], smoothing_function=chencherry.method1)
    print(f"BLEU score for candidate {i+1}: {score}")

# Initialize the Rouge scorer and calculate scores
rouge = Rouge()
rouge_scores = rouge.get_scores(candidates, [refs[0] for refs in reference_translations], avg=False)

# Extracting and printing ROUGE scores
for scores in rouge_scores:
    print(f"ROUGE-1: r: {scores['rouge-1']['r']:.4f}, p: {scores['rouge-1']['p']:.4f}, f: {scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2: r: {scores['rouge-2']['r']:.4f}, p: {scores['rouge-2']['p']:.4f}, f: {scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L: r: {scores['rouge-l']['r']:.4f}, p: {scores['rouge-l']['p']:.4f}, f: {scores['rouge-l']['f']:.4f}")

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



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
    [" This Earth Day, let's make every charge count for the planet! Introducing our Solar-Powered Chargers — the perfect companion for the eco-conscious tech enthusiast. Harness the power of the sun to keep your devices charged and your ecological footprint minimal. Ready to power your adventures sustainably?"],
    ["In an era where technology and environmental sustainability seem at odds, a new wave of innovation is proving otherwise. Sustainable gadgets are not just a trend; they're revolutionizing the tech industry, shifting paradigms towards a greener future. In this article, we explore five key ways these eco-friendly innovations are making their mark, from reducing waste to harnessing renewable energy. Join us as we delve into how sustainable gadgets are setting the new standard for the tech world."],
    ["Capture the world in stunning clarity while keeping the planet green with our EcoSmart Camera. Engineered with the environment in mind, this camera combines cutting-edge technology with eco-friendly features. Its durable, recycled casing reduces waste, while the low-energy consumption design ensures that your photographic adventures are both breathtaking and sustainable. Perfect for the eco-conscious photographer who does not want to compromise on quality or the planet’s health. With the EcoSmart Camera, every shot contributes to a greener tomorrow."],
    ["Dear EcoWarriors, This month, we're thrilled to bring you the latest in green technology innovations and exclusive offers from EcoFriendly Gadgets. As the world moves towards a more sustainable future, we're at the forefront, ensuring that our products not only elevate your lifestyle but also protect the planet. From groundbreaking eco-smart devices to our commitment to reducing electronic waste, we're excited to share how our innovations are contributing to a greener world. Plus, don't miss out on our special Earth Month discounts, available for a limited time. Join us in making sustainability the standard."]
]

# Get model translations
# candidates = get_model_translations('falcon:7b', source_texts)

candidates = ["Create a social post that promotes our new line of solar chargers and how they can help people reduce their carbon footprint. The post should be engaging and informative, and should include a call to action for people to learn more about our products and the benefits of using solar power.",
              "The tech industry is constantly evolving, and with it, the way we use and interact with technology. One of the most exciting developments in recent years has been the rise of sustainable gadgets. These devices are not only good for the environment, but they also offer a range of benefits for users. In this blog post, we'll explore five ways that sustainable tech is changing the tech landscape.1. Reduced Carbon Footprint: One major benefit of using sustainable technology is that it helps to reduce our carbon footprint. By using devices that are made from recycled materials or that use renewable energy sources, such as solar power, you can help to minimize your impact on the planet. This is especially important in the context of climate change, which is a major concern for many people.2. Improved Efficiency: Sustainable gadgets are often designed to be more efficient than their traditional counterparts. For example, many",
              "The Eco Smart Camera is the perfect solution for those looking to reduce their environmental impact while still enjoying the convenience of a security camera. This innovative device is designed to be energy efficient, using less power than traditional cameras and reducing your carbon footprint. With its sleek and modern design, the Eco Camera will blend seamlessly into any home or office environment. The camera is equipped with advanced motion detection technology, allowing it to only record when it detects movement, saving energy and storage space. It also features a built-in microphone and speaker, making it easy to communicate with visitors or monitor your home from anywhere. Whether you're looking for a way to save energy or simply want a high-quality security system, Eco smart camera has you covered.",
              "Dear Eco-Friendly Gadget Enthusiasts, We are excited to announce that we have some new and exciting innovations coming to our store this year! First, we are introducing a new line of eco-friendly gadgets that are sure to make your life easier and more sustainable. These gadgets include things like solar-powered lights, water-saving devices, and energy-efficient appliances. We also have a special offer for you this week: if you purchase any of our ecofriendly products, you will receive a free gift! This gift could be anything from a solar charger to a water bottle. So, what are you waiting for? Come check out our new products and take advantage of this great offer! Thank you for your support, Ecofriendly Gadgets"
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



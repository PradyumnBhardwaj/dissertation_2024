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
    ["To track your order status, please visit our website and go to the 'Orders' section under your account. There, you will find a list of your recent orders and a 'Track' button next to each order. Clicking on this button will provide you with the latest shipping information and estimated delivery date. If you need further assistance, feel free to contact our customer support team."],
    ["Para rastrear el estado de tu pedido, por favor visita nuestra página web y accede a la sección 'Pedidos' bajo tu cuenta. Allí encontrarás una lista de tus pedidos recientes y un botón de 'Rastrear' al lado de cada pedido. Al hacer clic en este botón, te proporcionaremos la información más reciente sobre el envío y la fecha estimada de entrega. Si necesitas más ayuda, no dudes en contactar a nuestro equipo de soporte al cliente."],
    ["Pour suivre l'état de votre commande, veuillez visiter notre site web et aller à la section 'Commandes' sous votre compte. Là, vous trouverez une liste de vos commandes récentes et un bouton 'Suivre' à côté de chaque commande. En cliquant sur ce bouton, vous obtiendrez les dernières informations sur l'expédition et la date de livraison estimée. Si vous avez besoin d'une aide supplémentaire, n'hésitez pas à contacter notre équipe de support client."]
]


# Get model translations
# candidates = get_model_translations('falcon:7b', source_texts)

candidates = ["You can track your order by clicking on the link in the email you received when your package was shipped. How long does it take to receive my package? We ship all orders within 24 hours of receiving them. Once your item is shipped, you will receive an email with a tracking number. Please allow 1-3 business days for your tracking information to update. How do I return an item?",
              "P* You can track your order on the website of the Russian Post. ... View More How long does it take to get to the US? I ordered on 12/11/2021 and it still hasn’t shipped I ordered 2 pairs of shoes and I only received one pair. I want my money back. What size is the 40?",
              "Vous recevrez un email de confirmation de votre commande. Vous recevrez également un autre email lorsque votre colis sera prêt à être expédié. Comment suivre mon colis? Vous pouvez suivre votre livraison en cliquant sur le lien dans l’email de suivi de commande que vous avez reçu. Si vous n’avez pas reçu cet email, veuillez nous contacter à l‘adresse suivante: info@the-art-of-living.com Où est mon produit? Veuillez noter que les délais de livraison peuvent varier en fonction de la destination. Veuillez nous noter si vous ne recevez pas votre produit dans les 14 jours suivant la date de l ‘expédition."
              ]

# Display Generated Text
for candidate in candidates:
    print("Generated Text:", candidate)

# Tokenize sentences for BLEU calculation
references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]

chencherry = SmoothingFunction()

print("GEMMA- Experiment-1")

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



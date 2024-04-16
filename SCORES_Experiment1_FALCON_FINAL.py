from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from rouge import Rouge
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import ollama
from evaluate import load

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def get_model_translations(model, messages):
    responses = []
    for message in messages:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': message}])
        responses.append(response['message']['content'])
    return responses

def calculate_ter(generated_texts, reference_texts):
    ter_scores = []
    for candidate, refs in zip(generated_texts, reference_texts):
        references = [[ref] for ref in refs]  
        ter = sacrebleu.corpus_ter([candidate], references)  
        ter_scores.append(ter.score)
    return ter_scores

bertscore = load("bertscore")

reference_translations = [
    ["Thank you for reaching out to us, and we apologize for the delay in your order. To provide you with an update, could you please share your order number? We're committed to ensuring your satisfaction and will promptly check the status of your order and provide you with an estimated delivery date. Your patience is greatly appreciated."],
    ["We're truly sorry to hear about the difficulty you've experienced in reaching our support team. We strive to be accessible to all our customers, and we acknowledge that at times, high demand can lead to longer wait times. Please let us know the best way to contact you or feel free to share your issue here, and we promise to address it as swiftly as possible."],
    ["We apologize that your order did not meet your expectations. Your satisfaction is our top priority. Please provide your order number, and we'll guide you through the process of returning the incorrect item and either replacing it with the correct one or processing a refund for you. Rest assured, we'll make this as seamless as possible for you."],
    ["We're sorry to hear your issue hasn't been resolved despite your previous reports. This isn't the experience we want for our customers. Could you please share your ticket or case number? We'll escalate this to our senior support team to ensure it's resolved promptly and keep you updated throughout the process."],
    ["It's disappointing to hear that your product arrived damaged. We understand how frustrating this can be and are here to help. Please share some photos of the damage, along with your order number, so we can initiate the process for a replacement or refund, depending on your preference. Your satisfaction is crucial to us."],
    ["We truly apologize for the inconvenience of having to explain your issue multiple times. To avoid any further repetition, we've documented your complaint in detail. Could you please confirm your case or order number so we can directly address and resolve your concern without further delay?"],
    ["We hear your frustration and apologize if our service has felt impersonal. We value your experience and would be glad to connect you with a member of our support team for a more personal touch. Could you please provide your preferred contact details and a suitable time for us to call you?"]
]

candidates = ["The tracking number for a package should be sent to you by email from either UPS or FedEx when your order is shipped. If you did not receive your email, please try the following: - Check your Spam/Junk folder. - If your order was paid via Paypal, the shipping confirmation email can be found under the Recent Transactions section of your Paypal account. The email should be from info@shop-cottoncandy.com. - Check your order confirmation email. ",
              "Hi @MisterG, Thank you for reaching out in the Community. I am sorry to hear that you are having trouble contacting our support team. We understand that this must be very frustrating. For this reason I have submitted a ticket on your behalf. Please be patient as we do our best to assist. We appreciate your patience.",
              "We will accept returns of unused products with a receipt. For a refund or exchange, we must receive the products in their original condition, unopened and unused with no damage to the packaging. For more details on returns and exchanges, read our return policy here.",
              "Hi @Lisamarie_L, I apologize for any confusion you're experiencing. As I understand you're having issues with the chat feature not connecting with users on the website. I've done some testing on your website and was unable to replicate the chat not connecting. I am able to connect and chat from the website. I'm sorry for any inconvenience you're experiencing.",
              "If you received a product that is damaged, first take some photos. Then contact us by email with your order #. How does the return process work? We accept returns of unworn, unused products, with tags still attached. The product must be returned within 30 days of delivery. Please contact us for return instructions and a prepaid UPS shipping label. I don't like the product I ordered. Can I get a refund? Unfortunately, we cannot offer a refund or exchange on any product. When will I receive my order? Orders are shipped Monday thru Friday.",
              "I'm frustrated because I've had to explain my issue multiple times. Can you help me without asking me to repeat everything again? It's not about me repeating myself, it's about me having to explain to you multiple times.",
              "If you have contacted our support team through our support chat or by email, the answer is YES! We have a team of live human beings who can assist you, and they are all here in the United States. Our support hours are Monday through Friday from 7:30 am - 6:00 pm. We have an office in Los Angeles as well as a call center in Denver. If you've contacted us through our online chat, you can always request a phone call or to be emailed back."
              ]

for candidate in candidates:
    print("Generated Text:", candidate)

references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]

chencherry = SmoothingFunction()

print("FALCON- Experiment-1")

for i, candidate_tokenized in enumerate(candidates_tokenized):
    score = corpus_bleu([references_tokenized[i]], [candidate_tokenized], smoothing_function=chencherry.method1)
    print(f"BLEU score for candidate {i+1}: {score}")


rouge = Rouge()
rouge_scores = rouge.get_scores(candidates, [refs[0] for refs in reference_translations], avg=False)


for i, scores in enumerate(rouge_scores):
    print(f"Candidate {i+1} ROUGE-1: r: {scores['rouge-1']['r']:.4f}, p: {scores['rouge-1']['p']:.4f}, f: {scores['rouge-1']['f']:.4f}")
    print(f"Candidate {i+1} ROUGE-2: r: {scores['rouge-2']['r']:.4f}, p: {scores['rouge-2']['p']:.4f}, f: {scores['rouge-2']['f']:.4f}")
    print(f"Candidate {i+1} ROUGE-L: r: {scores['rouge-l']['r']:.4f}, p: {scores['rouge-l']['p']:.4f}, f: {scores['rouge-l']['f']:.4f}")


ter_scores = calculate_ter(candidates, reference_translations)

for i, score in enumerate(ter_scores):
    print(f"Candidate {i+1} TER Score: {score:.2f}")

tokenized_candidates = [word_tokenize(cand) for cand in candidates]
tokenized_reference_translations = [[word_tokenize(ref) for ref in refs] for refs in reference_translations]

meteor_scores = [meteor_score(refs, cand) for cand, refs in zip(tokenized_candidates, tokenized_reference_translations)]

for i, score in enumerate(meteor_scores):
    print(f"METEOR Score for text {i+1}: {score:.4f}")
results = bertscore.compute(predictions=candidates, references=reference_translations, lang="en")

for i in range(len(candidates)):
    print(f"BERTScore Text {i+1} - Precision: {results['precision'][i]}, Recall: {results['recall'][i]}, F1 Score: {results['f1'][i]}")



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


candidates = ["I have been trying to contact you for a few days now and I have not received a response. I am very disappointed with your customer service. Can you please tell us when we can expect our order? We have waited for more than a month and we have still not heard from you. We are very frustrated and disappointed.",
              "I'm trying the chat and it says I have to wait 15 minutes. I wait and then it tells me to try again later. This is ridiculous. I have a problem with my account. It says that I need to verify my identity. When I click on the link to do so, it takes me back to the same page. How do I fix this? My account is locked and I can’t get in. What do i do",
              "We are sorry to hear that you are not satisfied with your order. Please contact us at info@the-art-of-living.com and we will be happy to assist you. I have a question about my purchase. Who can help me Please contact our customer service team at +31 20 7700105 or info @ the-artofliving .com. We are available Monday to Friday from 9:0am to 5:pm I want to return my product. What do I need to do? We accept returns within 14 days of receipt. The product must be in its original condition and packaging. To initiate a return, please contact customer support at 02 303 404 60 or email us. You will receive",
              "I have the same issue. I have reported it twice. Same issue here. Reported it once. No response. This is a known issue and we are working on a fix.  We will update this thread when we have more information.<",
              "If you received a damaged product, please contact us at support@the-shop.com. We will be happy to assist you with a refund or replacement. How long does it take to receive my order? We aim to ship all orders within 2-3 business days. Delivery times may vary depending on your location, but most orders should arrive within a week. If you have any questions or concerns about your order, don't hesitate to contact our customer service team.",
              "I have a 2013 Ford Escape with 100,025 miles. I have had the car for 3 years and have never had any issues. About 4 months ago, I started to notice a slight vibration in the steering wheel. It was very slight and only noticeable when I was driving on the highway. The vibration was not noticeable at all when driving around town. At first, it was only when the vehicle was cold. After driving for a while, the vibration would go away. However, over the past 6 weeks, this vibration has gotten worse. Now, even when my vehicle is warm, there is a vibration when going over 50 mph. When I am driving at 70-75 mph, my steering feels like it is shaking. This vibration is not constant.",
              "I've been trying to get a refund for a purchase I made on the app. I have been told to contact the seller, but the problem is that the person who sold me the item is no longer on Etsy. Can someone please help me? I just want my money back."
              ]

for candidate in candidates:
    print("Generated Text:", candidate)

references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]

chencherry = SmoothingFunction()

print("GEMMA- Experiment-1")

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



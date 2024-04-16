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
    ["If you've forgotten your password, please follow these steps to reset it: Go to the login page and click on the 'Forgot Password' link. Enter your email address associated with your account and submit the form. Check your email for a password reset link from us. Make sure to check your spam or junk folder if you don't see it in your inbox. Click on the link in the email, which will redirect you to a page where you can set a new password. If you encounter any issues during this process, please don't hesitate to contact our support team for further assistance."],
    ["If your device won't turn on even after charging, try these steps: Ensure that the charger and cable are correctly connected and the power outlet is functioning. Leave your device connected to the charger for at least 30 minutes. Try performing a hard reset on your device by holding down the power button for 10 seconds or following the specific hard reset procedure for your device model. If the device still does not turn on, there may be a hardware issue, and we recommend contacting our support team for a repair or replacement."],
    ["If you're having trouble connecting your device to Wi-Fi, please try the following troubleshooting steps: Restart your device and the Wi-Fi router. This can often resolve connection issues. Forget the Wi-Fi network on your device and then try reconnecting to it. You'll need to re-enter the network's password. Check if the issue is with a specific device or all devices. If it's all devices, the issue might be with the router or network. Ensure your device's software is up to date, as updates can include fixes for connectivity issues. If these steps don't resolve the issue, please reach out to our support team for further assistance."],
    ["If an application won't open following a software update freeze, try the following steps: Restart your device to see if that resolves the issue. Try opening the application in Safe Mode (if applicable) to see if it runs without the added configurations of the normal mode. Check for any available updates for the application or the device's operating system and install them. If the issue persists, consider uninstalling and reinstalling the application. Make sure to back up any important data before doing so to avoid loss. If these steps do not solve the problem, please contact our support team for assistance, ensuring to mention the steps already taken."],
    ["If your computer is not recognizing an external USB device: Try plugging the device into a different USB port on your computer. Restart your computer with the device connected. Update your computer's operating system and drivers, especially the USB controllers. Try the device with another computer to determine if the issue is with the computer or the device. If possible, check Device Manager (on Windows) or System Information (on Mac) to see if the device is listed under USB devices. If it's listed with a warning, you may need to update the driver manually. If none of these steps work, the issue might be with the USB device itself, or there may be a deeper hardware issue with your computer. Contacting our support team can provide further guidance."]
]


# Get model translations
# candidates = get_model_translations('falcon:7b', source_texts)

candidates = ["If you forget your password, you can reset it through your account. To reset your password: Click on My Account: My Account. Enter your email address. Click Send reset link. Follow the instructions to change your password. Note: This is not an email link. Please copy the link and use it in your browser.",
              "If your device won't turn on or won't charge, try these steps: Turn off your device by pressing and holding the Power/Lock key until the red slider appears. Then, slide to turn off your device. Plug your device into a wall outlet or a USB port (preferred), and try to power on or charge your device. If your device won't turn on at all, make sure you're pressing your device's Power/Lock key for 30 seconds or more.",
              "Make sure that you have the latest Wi-Fi settings for your device. If this doesn't work, try rebooting your device. If that doesn't work, try resetting the network settings of your device. If that still doesn't work, try resetting network and Internet options.",
              "If your app froze during the update, you should try updating again. If you're unable to open your app or access its content, you can try resetting or deleting it. After resetting or deleting, reinstall your app.",
              "If you have the device plugged into the back of your computer, try unplugging it and plugging it back in. If you have the device plugged into the front of your computer, unplug it and plug it back in there. Sometimes this helps the device become recognized properly. Also, it could be a driver issue. Check with the manufacturer of your computer for a driver update. If you still have issues after following both of these steps, we would recommend contacting our Tech Support team. They can be reached at. Please sign in to leave a comment."
              ]

# Display Generated Text
for candidate in candidates:
    print("Generated Text:", candidate)

# Tokenize sentences for BLEU calculation
references_tokenized = [[ref.split() for ref in refs] for refs in reference_translations]
candidates_tokenized = [cand.split() for cand in candidates]

chencherry = SmoothingFunction()
print("FALCON- Experiment-3")

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



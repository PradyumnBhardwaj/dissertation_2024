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
    ["If you've forgotten your password, please follow these steps to reset it: Go to the login page and click on the 'Forgot Password' link. Enter your email address associated with your account and submit the form. Check your email for a password reset link from us. Make sure to check your spam or junk folder if you don't see it in your inbox. Click on the link in the email, which will redirect you to a page where you can set a new password. If you encounter any issues during this process, please don't hesitate to contact our support team for further assistance."],
    ["If your device won't turn on even after charging, try these steps: Ensure that the charger and cable are correctly connected and the power outlet is functioning. Leave your device connected to the charger for at least 30 minutes. Try performing a hard reset on your device by holding down the power button for 10 seconds or following the specific hard reset procedure for your device model. If the device still does not turn on, there may be a hardware issue, and we recommend contacting our support team for a repair or replacement."],
    ["If you're having trouble connecting your device to Wi-Fi, please try the following troubleshooting steps: Restart your device and the Wi-Fi router. This can often resolve connection issues. Forget the Wi-Fi network on your device and then try reconnecting to it. You'll need to re-enter the network's password. Check if the issue is with a specific device or all devices. If it's all devices, the issue might be with the router or network. Ensure your device's software is up to date, as updates can include fixes for connectivity issues. If these steps don't resolve the issue, please reach out to our support team for further assistance."],
    ["If an application won't open following a software update freeze, try the following steps: Restart your device to see if that resolves the issue. Try opening the application in Safe Mode (if applicable) to see if it runs without the added configurations of the normal mode. Check for any available updates for the application or the device's operating system and install them. If the issue persists, consider uninstalling and reinstalling the application. Make sure to back up any important data before doing so to avoid loss. If these steps do not solve the problem, please contact our support team for assistance, ensuring to mention the steps already taken."],
    ["If your computer is not recognizing an external USB device: Try plugging the device into a different USB port on your computer. Restart your computer with the device connected. Update your computer's operating system and drivers, especially the USB controllers. Try the device with another computer to determine if the issue is with the computer or the device. If possible, check Device Manager (on Windows) or System Information (on Mac) to see if the device is listed under USB devices. If it's listed with a warning, you may need to update the driver manually. If none of these steps work, the issue might be with the USB device itself, or there may be a deeper hardware issue with your computer. Contacting our support team can provide further guidance."]
]


# Get model translations
# candidates = get_model_translations('falcon:7b', source_texts)

candidates = ["If you have forgotten your password, you can reset it by clicking on the Forgot your Password? link on our login page. You will be prompted to enter your email address, and we will send you an email with a link to reset your account. How can I change my email or password? You can change your login email and password by going to the Account Settings page and clicking the Edit button next to your current email. From there, enter the new email you would like to use and click Save. To change the password for your new account, click the Change Password button and enter a new password. How long does it take to receive my order? We aim to ship all orders within 24 hours of receiving them. However, please allow up to 48 hours for processing during peak periods. Once your order has been shipped, it should arrive within the estimated delivery time frame provided at",
              "If your device is not turning on after a full charge, please try the following: 1. Press and hold the power button for 10 seconds. 2. If the device does not turn back on or respond, try charging the battery for another 30 minutes. * If your battery is fully charged, but the screen is still blank, the display may be damaged. Please contact us for further assistance.< ",
              "1. Make sure that the Wi Fi network you are trying to join is not password protected. If it is, you will need to enter the password to access the Internet.2. Check to see if the device is in range of the router. The device must be within 100 feet of a router to establish a connection. * If the distance is too great, try moving the devices closer together. You may also want to consider purchasing a Wi‐Fi extender to increase the range.  3. Try connecting to a different WiFi router or network. This will help determine if there is an issue with the current network or if it's a problem with your device.",
              "I'm using a Mac. Thanks. Hi, Please try to run the following command in the terminal: <code>sudo rm -rf /Library/Application\ Support/Avid/Pro Tools/12.8.1/</code> Then try launching Pro Tools again. If it still doesn' t work, please try the same command but with 13.0.2 instead of <code>11.3</code>.<",
              "<strong>Answer:</strong> 1. Make sure the device is plugged into a USB port. 2. If the USB drive is a flash drive, make sure it is inserted all the way into the port and that the cap is on the end of the drive. 3. Restart your computer. 4. Check the Device Manager to see if the computer recognizes the external device. To do this, click on <strong>Start</strong>, then <em>Control Panel</em>, and then click <b>Device Manager</b>. Look for the <i>Universal Serial Bus controllers</i> section. The USB devices should be listed under this section, but if they are not, the problem may be with the hardware."
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



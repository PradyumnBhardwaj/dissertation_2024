from rouge import Rouge
import ollama

response = ollama.chat(model='falcon:7b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
generated_text = response['message']['content']
print(generated_text)

rouge = Rouge()
reference_text = "The sky appears blue because of the way Earth's atmosphere absorbs and reflects sunlight. When the sun's light enters the Earth's atmosphere, it is scattered by molecules in the air, causing some of the light to be absorbed and others to be reflected back out into space. The light that is reflected back to our planet is mostly blue, which is why the sky appears blue."

scores = rouge.get_scores(generated_text, reference_text)

rouge_1_scores = scores[0]['rouge-1']
rouge_2_scores = scores[0]['rouge-2']
rouge_l_scores = scores[0]['rouge-l']
print(f"ROUGE-1: r: {rouge_1_scores['r']:.4f}, p: {rouge_1_scores['p']:.4f}, f: {rouge_1_scores['f']:.4f}")
print(f"ROUGE-2: r: {rouge_2_scores['r']:.4f}, p: {rouge_2_scores['p']:.4f}, f: {rouge_2_scores['f']:.4f}")
print(f"ROUGE-L: r: {rouge_l_scores['r']:.4f}, p: {rouge_l_scores['p']:.4f}, f: {rouge_l_scores['f']:.4f}")

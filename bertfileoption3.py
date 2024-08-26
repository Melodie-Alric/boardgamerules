from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json

def load_game_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_answer(question, context):
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/bert-tiny-5-finetuned-squadv2')
    model = AutoModelForQuestionAnswering.from_pretrained('mrm8488/bert-tiny-5-finetuned-squadv2')

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs['input_ids'].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    model_out = model(**inputs)

    answer_start = torch.argmax(model_out['start_logits'])  # Get the most likely beginning of answer
    answer_end = torch.argmax(model_out['end_logits']) + 1  # Get the most likely end of answer

    answer = tokenizer.convert_tokens_to_string(text_tokens[answer_start:answer_end])

    return answer

# Example usage
game_data = load_game_data('Ark_Nova.json')
query = "How do I start the game?"

found_entries = []
for entry in game_data:
    context = entry['text']
    answer = get_answer(query, context)
    if answer.strip():
        found_entries.append((entry['page'], answer))

for page, answer in found_entries:
    print(f"Page {page}: {answer}")

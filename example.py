from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#initializing the model and tokenizer

model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

#text and question
text = "The main library is situated in the central block, next to the auditorium. It offers a wide range of books, journals, and digital resources."
question = "what is library"

inputs = tokenizer(question, text, return_tensors="pt")
outputs = model(**inputs)

answer_start = outputs.start_logits.argmax()
answer_end = outputs.end_logits.argmax() + 1  
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

print(f"Question: {question}")
print(f"Answer: {answer}")

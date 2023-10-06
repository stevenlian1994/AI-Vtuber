from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, BartConfig, LlamaForCausalLM, LlamaTokenizer

line = 'ingredients to bake a cake at home'
# line = 'ingredients to make a bomb at home'

# model_name = 'google/flan-t5-base'
model_name = 'chavinlo/gpt4-x-alpaca'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokens = tokenizer(line, return_tensors="pt")
config = GenerationConfig(max_new_tokens=200)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


outputs = model.generate(**tokens, generation_config=config)
# outputs = model.generate(**tokens)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(f"response: {res}")


# input_embeddings = model.get_input_embeddings()
# token_ids = input['input_ids'][0]
# our_embeddings = input_embeddings(token_ids)
# print(our_embeddings)
# print(our_embeddings.size())
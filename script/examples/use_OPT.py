from transformers import GPT2Tokenizer, OPTForCausalLM
import os

# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

model = OPTForCausalLM.from_pretrained("facebook/opt-30b")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-30b")

# prompt = "Hey, are you consciours? Can you talk to me?"
prompt = [
    "This is a news about sports: ",
    "This is a news about politics: ",
    "This is a news about IBM PC: ",
    "This is a news about electronics: ",

]
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate
generate_ids = model.generate(**inputs, max_length=30)
generate_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for text in generate_text:
    print(text)

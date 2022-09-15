import os
import openai

openai.api_key = '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'

for i in range(10):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Protein1 and protein2 are related:",
        temperature=0.7,
        max_tokens=32,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    text = response['choices'][0]['text'].strip()
    print(text)

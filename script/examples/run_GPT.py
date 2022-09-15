import os
import openai

openai.api_key = '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'


label_words = [
    ['hockey'], ['baseball'], ['guns'], ['crypt'], ['electronics'], ['mac'], ['motorcycles'],
    ['mideast'], ['atheism'], ['microsoft windows'], ['automobiles'], ['medicine'], ['christian'], ['IBM'],
    ['sale'], ['politics'], ['windows x'], ['space'], ['graphics'], ['religion']
]
label_words = [
    ['negative'], ['somewhat negative'], ['neutral'], ['somewhat positive'], ['positive']
]
for label_word in label_words:
    w = label_word[0]
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Write a sentence with {w} sentiment:",
        temperature=0.7,
        max_tokens=32,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    text = response['choices'][0]['text'].strip()
    print(w)
    print(text)

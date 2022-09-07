from openprompt.data_utils import InputExample
import sys

sys.path.append('../..')

label_words = [
    ['hockey'], ['baseball'], ['guns'], ['crypt'], ['electronics'], ['mac'], ['motorcycles'],
    ['mideast'], ['atheism'], ['ms-windows'], ['automobiles'], ['medicine'], ['christian'], ['ibm'],
    ['sale'], ['politics'], ['windows x'], ['space'], ['graphics'], ['religion']
]

dataset = [  # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid=0,
        text_a=f"This is a news about {label_word[0]}.",
    ) for label_word in label_words
]
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

from openprompt.prompts import ManualTemplate

promptTemplate = ManualTemplate(
    text='The topic of news: {"mask"}. {"placeholder":"text_a"}',
    tokenizer=tokenizer,
)
from openprompt.prompts import ManualVerbalizer

promptVerbalizer = ManualVerbalizer(
    classes=list(range(20)),
    label_words=label_words,
    tokenizer=tokenizer,
)
from openprompt import PromptForClassification

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)

wrapped_tokenizer = WrapperClass(max_seq_length=256, tokenizer=tokenizer, truncate_method="tail")
wrapped_example = promptTemplate.wrap_one_example(dataset[0])
print(wrapped_example)

tokenized_example = wrapped_tokenizer.tokenize_one_example(
    wrapped_example, teacher_forcing=False
)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))

from openprompt import PromptDataLoader

data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)
import torch

# making zero-shot inference using pretrained MLM with prompt
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(logits, preds)
# predictions would be 1, 0 for classes 'positive', 'negative'

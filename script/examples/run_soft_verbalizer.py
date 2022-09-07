from openprompt.data_utils.text_classification_dataset import AgnewsProcessor
from openprompt.data_utils import InputExample
from sklearn.metrics import confusion_matrix

dataset = {}
dataset['train'] = AgnewsProcessor().get_train_examples("TextClassification/agnews")
# We sample a few examples to form the few-shot training pool
from openprompt.data_utils.data_sampler import FewShotSampler

sampler = FewShotSampler(num_examples_per_label=128, num_examples_per_label_dev=16, also_sample_dev=True)
dataset['train'], dataset['validation'] = sampler(dataset['train'])
tmp = []
for x in dataset['train']:
    if x.label == 0:
        tmp.append(x)
dataset['train'] = tmp
dataset['test'] = AgnewsProcessor().get_test_examples("TextClassification/agnews")

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")

from openprompt.prompts import ManualTemplate

mytemplate = ManualTemplate(tokenizer=tokenizer,
                            text='{"placeholder":"text_a"} {"placeholder":"text_b"} In this sentence, the topic is {"mask"}.')

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                    batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")
# next(iter(train_dataloader))

# ## Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

from openprompt.prompts import SoftVerbalizer, ProtoVerbalizer, ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
# myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=4,
#          label_words=["politics", "sports", "business", "technology"])
myverbalizer = ProtoVerbalizer(tokenizer, plm, num_classes=4,
                               label_words=["politics", "sports", "business", "technology"])
# myverbalizer = ManualVerbalizer(tokenizer, num_classes=4,
#                                 classes=["politics", "sports", "business", "technology"],
#                                 label_words=["politics", "sports", "business", "technology"])
# or without label words
# myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=4)
# myverbalizer = ProtoVerbalizer(tokenizer, plm, num_classes=4)

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

proto_dataset = [  # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid=0,
        text_a="This is a news about politics.",
        text_b=""
    ),
    InputExample(
        guid=1,
        text_a="This is a news about sports.",
        text_b=""
    ),
    InputExample(
        guid=2,
        text_a="This is a news about business.",
        text_b=""
    ),
    InputExample(
        guid=3,
        text_a="This is a news about technology.",
        text_b=""
    ),

]
proto_dataloader = PromptDataLoader(dataset=proto_dataset, template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                    batch_size=1, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")
print('***testing proto***')
with torch.no_grad():
    for batch in proto_dataloader:
        if use_cuda:
            batch = batch.cuda()

        logits = prompt_model(batch)
        logits = logits.exp() / logits.exp().sum()
        print(logits)
        preds = torch.argmax(logits, dim=-1)

# ## below is standard training


from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters

optimizer_grouped_parameters2 = [
    {'params': prompt_model.verbalizer.group_parameters_proto, "lr": 3e-5},
]

optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
optimizer2 = AdamW(optimizer_grouped_parameters2)

for epoch in range(5):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        print(tot_loss / (step + 1))

# ## evaluate

# %%
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                         batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                         truncate_method="head")

prompt_model.eval()

allpreds = []
alllabels = []
for step, inputs in enumerate(validation_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
print(confusion_matrix(alllabels, allpreds))
acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
print("validation:", acc)

# test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
#                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
#                                    batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
#                                    truncate_method="head")
#
# allpreds = []
# alllabels = []
# for step, inputs in enumerate(test_dataloader):
#     if use_cuda:
#         inputs = inputs.cuda()
#     logits = prompt_model(inputs)
#     labels = inputs['label']
#     alllabels.extend(labels.cpu().tolist())
#     allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
# acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
# print("test:", acc)  # roughly ~0.85

with torch.no_grad():
    for batch in proto_dataloader:
        if use_cuda:
            batch = batch.cuda()

        logits = prompt_model(batch)
        logits = logits.exp() / logits.exp().sum()
        print(logits)
        preds = torch.argmax(logits, dim=-1)

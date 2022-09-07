from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM


class Template:

    def __init__(self, ls, template_text):
        self.mixed_token_start = "{"
        self.mixed_token_end = "}"
        self.template_text = template_text
        self.text = [self.parse_text(t) for t in ls]

    @staticmethod
    def incorporate_text_example(text: List[Dict], replacement: Dict) -> str:
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d["add_prefix_space"] + replacement[d['placeholder']]
            elif 'soft' in d:
                text[i] = ''
            elif 'mask' in d:
                text[i] = '[MASK]'
            elif 'sep' in d:
                text[i] = '[SEP]'
            elif 'special' in d:
                text[i] = d['special']
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return ' '.join(text)

    def parse_text(self, replacement: dict) -> str:
        text = self.template_text
        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix_space": ' ' if (i > 0 and text[i - 1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix_space"] = ' '
                i = i + 1
            if i == len(text):
                break

            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j].rstrip(' ')
                i = j

            else:
                j = i + 1
                mixed_token_cnt = 1  # { {} {} } nested support
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        mixed_token_cnt -= 1
                        if mixed_token_cnt == 0: break
                    elif text[j] == self.mixed_token_start:
                        mixed_token_cnt += 1
                    j = j + 1
                if j == len(text):
                    raise ValueError(
                        f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{' + text[i + 1:j] + '}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    import traceback
                    print(traceback.format_exc())
                    print(f"syntax error in {dict_str}")
                    exit()
                i = j + 1

            parsed.append(d)
        return self.incorporate_text_example(parsed, replacement)


# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# template = 'The topic of news: {"mask"}. {"sep"} {"placeholder":"text_a"}'
# texts = [{"text_a": "This is a sport news"},
#          {"text_a": "This a politics news"}]
# tmp = Template(texts, template)
# text = tmp.text
# tokenized_text = tokenizer(text)
# print(tokenized_text["input_ids"])

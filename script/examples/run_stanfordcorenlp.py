import os.path

from stanfordcorenlp import StanfordCoreNLP

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

base_dir = os.path.expanduser('~/src')

nlp = StanfordCoreNLP(os.path.join(base_dir, 'stanford-corenlp-4.4.0'))
# nlp = StanfordCoreNLP('http://localhost', port=9000)  #通过服务器访问

sentence = 'The results show that Protein1 and Protein2 form a complex and suggest that they participate in epithelial cell differentiation in the developing kidney.'

print('Tokenize:', nlp.word_tokenize(sentence))  # 令牌化
print('Part of Speech:', nlp.pos_tag(sentence))  # 词性标注
print('Named Entities:', nlp.ner(sentence))  # 命名实体
print('Constituency Parsing:', nlp.parse(sentence))  # 语法树，成分句法把句子组织成短语的形式
print('Dependency Parsing:', nlp.dependency_parse(sentence))  # 依存句法 揭示句子中词的依赖关系

props = {'timeout': '5000000', 'annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true',
         'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
results = nlp.annotate(' '.join(sentence), properties=props)

nlp.close()

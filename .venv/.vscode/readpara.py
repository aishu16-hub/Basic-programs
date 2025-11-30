import nltk
from nltk import word_tokenize

# Download required resources once
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Read paragraph from file
with open('/home/aishwarya/intel-training/.venv/.vscode/paragraph.txt', 'r') as f:
    paragraph = f.read()

# Tokenize and POS tagging
tokens = word_tokenize(paragraph)
pos_tags = nltk.pos_tag(tokens)

# Extract nouns and verbs
nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
verbs = [word for word, pos in pos_tags if pos.startswith('VB')]

print('Nouns:', nouns)
print('Verbs:', verbs)

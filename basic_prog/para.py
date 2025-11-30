import spacy

# Load English tokenizer, POS tagger, etc.
nlp = spacy.load('en_core_web_sm')

# Read the paragraph from a file
with open('paragraph.txt', 'r') as file:
    text = file.read()

# Process the text
doc = nlp(text)

# Extract nouns and verbs
nouns = [token.text for token in doc if token.pos_ == 'NOUN']
verbs = [token.text for token in doc if token.pos_ == 'VERB']

print("Nouns:", nouns)
print("Verbs:", verbs)


with open('paragraph.txt', 'r') as file:
    text = file.read()
print('File content:', text)


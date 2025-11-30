import spacy
#The spaCy model parses the paragraph and identifies parts of speech for each token (word)

nlp = spacy.load("en_core_web_sm")

# Instead of reading from a file, directly assign the paragraph here
paragraph = (
    "In the quiet town of Willowbrook, the autumn leaves fell gently to the ground, "
    "creating a colorful mosaic along the winding paths. Children played happily in the crisp air, "
    "their laughter echoing through the streets. Nearby, an elderly man tended to his garden, "
    "carefully pruning the roses that bloomed despite the approaching frost. The community thrived "
    "on a shared spirit of kindness and tradition, making Willowbrook a place where everyone felt at home."
)

doc = nlp(paragraph)

nouns = [token.text for token in doc if token.pos_ == "NOUN"]
verbs = [token.text for token in doc if token.pos_ == "VERB"]

print("Nouns:", nouns)
print("Verbs:", verbs)


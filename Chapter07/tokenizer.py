import nltk
nltk.download('punkt')

text = "Are you curious about tokenization? Let's see how it works! We need to analyze a couple of sentences with punctuations to see it in action."

# Sentence tokenization
from nltk.tokenize import sent_tokenize

sent_tokenize_list = sent_tokenize(text)
print("Sentence tokenizer:")
print(sent_tokenize_list)

# Create a new word tokenizer
from nltk.tokenize import word_tokenize

print("Word tokenizer:")
print(word_tokenize(text))

# Create a new WordPunct tokenizer
from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()
print("Word punct tokenizer:")
print(word_punct_tokenizer.tokenize(text))


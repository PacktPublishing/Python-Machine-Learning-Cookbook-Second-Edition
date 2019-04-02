import spacy

nlp = spacy.load('en_core_web_sm')

Text = nlp(u'We catched fish, and talked, and we took a swim now and then to keep off sleepiness')

for token in Text:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

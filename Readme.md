## Natural Language Processing

### 1. Tokenization
- It is a process that convert paragraphs/sentences into tokens

paragraphs ----Tokenize-----> Sentences ----Tokenize-----> Words/Tokens

Libraries used in Tokenizations are Nltk, Spacy


### 2. Stemming

Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or the roots of words known as a lemma . Stemming is important in natural language understanding and Nlp

### 3. Lemmatization
Lemmatization technique is like stemming. The output we will get after lemmatization is called lemma which is root word rather than root stem, the output stem. After lemmitization we will getting a valid word that means the same thing

### 4. Stopwords
Stopwords are the common words in a language that usually don’t carry significant meaning in text analysis.

Examples in English:
`the, is, in, at, of, a, and, to, for, with, on, by, it`

### 5.POS Tagging (Part-of-Speech Tagging)

POS tagging is the process of labeling each word in a sentence with its grammatical role (part of speech) such as noun, verb, adjective, adverb, pronoun, preposition, conjunction, etc.

Example:
Sentence: "The quick brown fox jumps over the lazy dog."
POS tagging result:

The → Determiner (DT)

quick → Adjective (JJ)

brown → Adjective (JJ)

fox → Noun (NN)

jumps → Verb (VBZ)

over → Preposition (IN)

the → Determiner (DT)

lazy → Adjective (JJ)

dog → Noun (NN)

So, POS tagging tells us the syntactic category of each word.

**Why useful?**

Helps in parsing sentences

Improves information retrieval and text classification

Used in machine translation and speech recognition

### 6.Named Entity Recognition (NER)

NER is the process of identifying and classifying real-world entities (like names, locations, organizations, dates, percentages, money, etc.) in text.

Example:
Sentence: "Barack Obama was born in Hawaii on August 4, 1961."
NER result:

Barack Obama → PERSON

Hawaii → LOCATION

August 4, 1961 → DATE

So, NER extracts semantic meaning related to entities.

**Why useful?**

Helps in question answering systems

Used in chatbots and voice assistants

Useful in resume parsing, medical text mining, and business intelligence
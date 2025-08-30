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


### 7.What is One Hot Encoding?

One Hot Encoding is a method to convert **categorical data (labels/text)** into a numerical **binary vector** so that machine learning models can understand it.

* Each category is represented as a vector of `0`s and a single `1`.
* The position of the `1` indicates the category.

---

#### Example

Suppose you have a column **Colors**:

```
Colors = [Red, Green, Blue, Green]
```

Unique categories = `{Red, Green, Blue}`

### One Hot Encoding result:

| Color | Red | Green | Blue |
| ----- | --- | ----- | ---- |
| Red   | 1   | 0     | 0    |
| Green | 0   | 1     | 0    |
| Blue  | 0   | 0     | 1    |
| Green | 0   | 1     | 0    |

---

### Why use it?

* ML models (like Linear Regression, Logistic Regression, Neural Networks) need **numbers** as input, not text.
* One Hot Encoding removes any **ordinal meaning** (e.g., "Red=1, Green=2, Blue=3" would incorrectly imply order).

---

### 8.What is Bag of Words (BoW)?

**Bag of Words** is a simple and commonly used **text representation technique** in Natural Language Processing (NLP).

* It converts text (sentences/documents) into **numerical vectors**.
* It only cares about **word occurrence (frequency or presence)**, not grammar or order.
* The "bag" means we just collect all words, ignoring sequence.

---

### Example

Suppose we have 2 sentences:

1. `"I love NLP"`
2. `"I love Machine Learning"`

### Step 1: Build Vocabulary (all unique words)

```
["I", "love", "NLP", "Machine", "Learning"]
```

### Step 2: Represent each sentence as a vector

| Sentence                  | I | love | NLP | Machine | Learning |
| ------------------------- | - | ---- | --- | ------- | -------- |
| "I love NLP"              | 1 | 1    | 1   | 0       | 0        |
| "I love Machine Learning" | 1 | 1    | 0   | 1       | 1        |

* `1` = word is present
* `0` = word is absent
* (Sometimes we also use **word counts** instead of 0/1)

---

### Variants of BoW

1. **Binary BoW** → Only presence/absence (0 or 1).
2. **Count BoW** → Word frequencies (how many times each word occurs).
3. **TF-IDF (Term Frequency - Inverse Document Frequency)** → A weighted BoW that reduces importance of common words (like *the, is, and*).

---


### Limitations of BoW

* Ignores **word order** (“dog bites man” vs “man bites dog” look the same).
* Ignores **context/semantics** (meaning).
* High **dimensionality** if vocabulary is large.

---


### 9.What is an N-Gram?

An **N-Gram** is a **sequence of N words (or tokens)** taken from a text.

* **Unigram (1-gram):** single words
* **Bigram (2-gram):** pairs of consecutive words
* **Trigram (3-gram):** three consecutive words
* And so on...

It helps capture some **context and order**, unlike Bag of Words (BoW).

---

### Example

Sentence:

```
"I love Natural Language Processing"
```

* **Unigrams (1-grams):**
  `["I", "love", "Natural", "Language", "Processing"]`

* **Bigrams (2-grams):**
  `["I love", "love Natural", "Natural Language", "Language Processing"]`

* **Trigrams (3-grams):**
  `["I love Natural", "love Natural Language", "Natural Language Processing"]`

---

### Why N-Grams?

* **Unigrams** ignore context (like BoW).
* **Bigrams/Trigrams** add local context by considering word sequences.
* Used in **language models, text classification, sentiment analysis, autocomplete, and speech recognition.**

---

### Limitations

* As **N increases**, vocabulary size grows very fast (sparse representation).
* Still limited in capturing **long-term dependencies**.
* Modern NLP uses **word embeddings + transformers (BERT, GPT)** for richer context.

---


# Understanding Word Vectors vs. Sentence Vectors

## Introduction

In Natural Language Processing (NLP), embeddings are a powerful tool used to represent text data in a numerical form. These embeddings can be at the word level (word vectors) or the sentence level (sentence vectors). Understanding the difference between these two types of embeddings is essential, as they serve different purposes and capture different aspects of language.

### Word Vectors
Word vectors are numerical representations of individual words. They capture semantic relationships between words, meaning that similar words have similar vectors. However, when it comes to representing sentences, simple aggregation of word vectors (e.g., by averaging) has limitations.

**Key Point:** Averaging word vectors to create sentence embeddings ignores the order of words and their context, potentially leading to inaccurate representations of the sentence’s meaning.

### Sentence Vectors
Sentence vectors, on the other hand, are embeddings that represent entire sentences. They capture not just the meaning of the individual words but also their order and the overall context, leading to a more accurate representation of the sentence's meaning.

**Key Point:** Sentence vectors preserve the context and the order of words, making them more suitable for tasks that require understanding the full meaning of sentences.

## Annotated Code Examples

### 1. Generating Word Vectors and Averaging to Create Sentence Embeddings

In this example, we generate word vectors for each word in a sentence and then average them to create a sentence embedding. This method, however, loses the order and context of the words.

```python
# Sample sentences
in_1 = "The kids play in the park."
in_2 = "The play was for kids in the park."

# Remove stop words
in_pp_1 = ["kids", "play", "park"]
in_pp_2 = ["play", "kids", "park"]

# Generate word embeddings
embeddings_1 = [emb.values for emb in embedding_model.get_embeddings(in_pp_1)]
embeddings_2 = [emb.values for emb in embedding_model.get_embeddings(in_pp_2)]

# Convert to 2D arrays
import numpy as np
emb_array_1 = np.stack(embeddings_1)
emb_array_2 = np.stack(embeddings_2)

# Average embeddings to create sentence vectors
emb_1_mean = emb_array_1.mean(axis=0) 
emb_2_mean = emb_array_2.mean(axis=0)

# Result: Two identical sentence embeddings despite different meanings
print(emb_1_mean[:4])
print(emb_2_mean[:4])
```

**Explanation:**
- **Stop Words Removal:** We remove common stop words to focus on the main words of the sentences.
- **Generating Embeddings:** We create word embeddings for the main words.
- **Averaging Embeddings:** We take the average of the word embeddings to produce a single sentence vector.
- **Result:** Despite the sentences having different meanings, the resulting sentence vectors are identical because the word order and context are ignored.

### 2. Generating Context-Aware Sentence Embeddings

In this example, we use a pre-trained model to generate sentence vectors that preserve the context and order of words, resulting in different embeddings for sentences with different meanings.

```python
# Sample sentences
in_1 = "The kids play in the park."
in_2 = "The play was for kids in the park."

# Generate sentence embeddings using a context-aware model
embedding_1 = embedding_model.get_embeddings([in_1])
embedding_2 = embedding_model.get_embeddings([in_2])

# Extracting the vectors
vector_1 = embedding_1[0].values
vector_2 = embedding_2[0].values

# Result: Two different sentence embeddings reflecting different meanings
print(vector_1[:4])
print(vector_2[:4])
```

**Explanation:**
- **Context-Aware Embeddings:** We use a model that generates embeddings for the entire sentence, taking into account the order of words and the overall context.
- **Result:** The two sentences, though similar in word choice, produce different sentence vectors because the model understands their different meanings.

## Conclusion

The choice between word vectors and sentence vectors depends on the specific application. If the goal is to understand individual words, word vectors suffice. However, for tasks requiring comprehension of entire sentences, context-aware sentence vectors are essential.

The code examples above demonstrate how word vectors might fail to capture the full meaning of sentences, while context-aware sentence vectors provide a more accurate representation. Understanding these differences is crucial for building effective NLP models.

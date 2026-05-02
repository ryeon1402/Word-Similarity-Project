from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

stopwords = {
    "is",
    "a",
    "the",
    "and",
    "of",
    "to",
    "in",
    "that",
    "with",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "for",
    "on",
    "as",
    "at",
    "by",
    "an",
    "or",
    "from",
    "but",
    "not",
    "can",
    "will",
    "just",
    "into",
    "than",
    "then",
    "too",
    "very",
}

with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

sentences = text.strip().split(".")

tokenized_sentences = []
for sentence in sentences:
    sentence = sentence.strip().lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    if sentence:
        words = sentence.split()
        filtered_words = [word for word in words if word not in stopwords]
        if filtered_words:
            tokenized_sentences.append(filtered_words)

word_counts = defaultdict(int)
for words in tokenized_sentences:
    for word in words:
        word_counts[word] += 1

filtered_sentences = []
for words in tokenized_sentences:
    filtered_words = [word for word in words if word_counts[word] >= 1]
    if filtered_words:
        filtered_sentences.append(filtered_words)

context_dict = defaultdict(list)
window_size = 3

for words in filtered_sentences:
    for i, target_word in enumerate(words):
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)

        for j in range(start, end):
            if i != j:
                context_dict[target_word].append(words[j])

words_list = []
documents = []

for word, context_words in context_dict.items():
    words_list.append(word)
    documents.append(" ".join(context_words))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)


def find_similar_words(target_word, top_n=5):
    if target_word not in words_list:
        return []

    target_index = words_list.index(target_word)
    target_vector = tfidf_matrix[target_index]

    similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()

    results = []
    for i, score in enumerate(similarities):
        if words_list[i] != target_word:
            results.append((words_list[i], score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


target = input("Enter a word: ").lower().strip()
results = find_similar_words(target)

if results:
    print(f"\nWords similar to '{target}':")
    for word, score in results:
        print(f"{word}: {score:.3f}")
else:
    print("Word not found.")

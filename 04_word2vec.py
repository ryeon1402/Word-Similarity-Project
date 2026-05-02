import re
from gensim.models import Word2Vec

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

model = Word2Vec(
    sentences=tokenized_sentences, vector_size=50, window=2, min_count=1, workers=4
)

target = input("Enter a word: ").lower().strip()

if target in model.wv:
    results = model.wv.most_similar(target, topn=5)

    print(f"\nWords similar to '{target}': ")
    for word, score in results:
        print(f"{word}: {score:.3f}")
else:
    print("Word not found.")

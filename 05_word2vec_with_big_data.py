from gensim.models import Word2Vec
from datasets import load_dataset
import re

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

with open("big_data.txt", "w", encoding="utf-8") as f:
    for split in ["train", "validation", "test"]:
        for line in ds[split]["text"]:
            if line.strip():
                f.write(line + "\n")

with open("big_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = text.split("\n")

tokenized_sentences = []

for sentence in sentences:
    sentence = sentence.strip().lower()

    sentence = re.sub(r"[^a-z\s]", "", sentence)

    words = sentence.split()

    if len(words) >= 2:
        tokenized_sentences.append(words)

model = Word2Vec(
    sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4
)

print("Training complete!")
print(len(model.wv))
print(list(model.wv.index_to_key)[:20])

while True:
    target = input("Enter a word: ").lower().strip()

    if target == "exit":
        break

    if target in model.wv:
        results = model.wv.most_similar(target, topn=10)
        print(f"\nWords similar to '{target}':")
        for word, score in results:
            print(f"{word}: {score:.3f}")
    else:
        print("Word not found.")

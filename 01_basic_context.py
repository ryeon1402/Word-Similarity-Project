from collections import defaultdict
import math

sentences = [
    "cat is a cute animal",
    "dog is a friendly animal",
    "cat and dog are pets",
    "apple and banana are fruits",
    "orange is a sweet fruit",
    "banana is a yellow fruit",
    "dog likes people",
    "cat likes fish",
]

stopwords = {"is", "a", "the", "and", "of", "to", "in", "that", "with", "are"}

tokenized_sentences = [
    [word for word in sentence.lower().split() if word not in stopwords]
    for sentence in sentences
]  # make the word as lowercase and split

context_dict = defaultdict(list)  # dict which saves neigbour words
window_size = 2  # check 2words in front and back

for words in tokenized_sentences:
    for i, target_word in enumerate(
        words
    ):  # i = index(location) target_word = real word
        start = max(0, i - window_size)  # check 2words in front and back
        end = min(
            len(words), i + window_size + 1
        )  # in python end point is not included so we need to add 1

        for j in range(start, end):  # except itself save neigbour words
            if i != j:
                context_dict[target_word].append(words[j])

word_vectors = {}  # making word vector

for (
    word,
    context_words,
) in (
    context_dict.items()
):  # count number of appearances of neigbour words for each word
    freq_dict = defaultdict(int)
    for context_word in context_words:
        freq_dict[context_word] += 1
    word_vectors[word] = freq_dict


def cosine_similarity(vec1, vec2):
    all_words = set(vec1.keys()) | set(vec2.keys())  # set without overlap + | = union

    dot_product = 0
    norm1 = 0
    norm2 = 0

    for word in all_words:
        v1 = vec1.get(word, 0)  # no word -> 0
        v2 = vec2.get(word, 0)

        dot_product += v1 * v2
        norm1 += v1**2
        norm2 += v2**2

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (
        math.sqrt(norm1) * math.sqrt(norm2)
    )  # close to 1 = similar, close to 0 no relationship


def find_similar_words(target_word, top_n=3):
    if target_word not in word_vectors:
        return []

    similarities = []

    for word in word_vectors:  # compare all words in dict
        if word != target_word:
            score = cosine_similarity(word_vectors[target_word], word_vectors[word])
            similarities.append((word, score))

    similarities.sort(key=lambda x: x[1], reverse=True)  # sort from highest to lowest
    return similarities[:top_n]  # return top3


target = input("Enter a word: ").lower()  # put word
results = find_similar_words(target)

if results:
    print(f"\nWord similar to '{target}': ")
    for word, score in results:
        print(f"{word}: {score:.3f}")  # 3 decimal point
else:
    print("Word not found.")

# CODTECH Internship Project
# Runtime Text Summarization Tool using NLP

import nltk
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, summary_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    word_freq = {}

    for word in words:
        if word.isalnum() and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    summary = heapq.nlargest(summary_sentences, sentence_scores, key=sentence_scores.get)
    return " ".join(summary)


# ---------------- USER INPUT ----------------
print("\nEnter the article text below (press ENTER twice to finish):\n")

user_text = ""
while True:
    line = input()
    if line == "":
        break
    user_text += line + " "

# ---------------- OUTPUT ----------------
if len(user_text.strip()) == 0:
    print("\nNo text entered!")
else:
    result = summarize_text(user_text)
    print("\n--- SUMMARY ---\n")
    print(result)

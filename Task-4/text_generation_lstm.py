import torch
import torch.nn as nn
import numpy as np

text = """
Artificial intelligence is transforming the world.
Machine learning allows systems to learn from data.
Deep learning uses neural networks with many layers.
AI is used in healthcare education and cybersecurity.
Technology is evolving rapidly in modern society.
"""

words = text.lower().replace('.', '').split()
vocab = sorted(set(words))

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

seq_length = 3
X = []
y = []

for i in range(len(words) - seq_length):
    X.append([word_to_ix[w] for w in words[i:i + seq_length]])
    y.append(word_to_ix[words[i + seq_length]])

X = torch.tensor(X)
y = torch.tensor(y)

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = TextLSTM(len(vocab), 50, 100)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nTraining the model...\n")
for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item()}")

def generate_text(seed_text, num_words):
    model.eval()
    words_list = seed_text.lower().split()

    for _ in range(num_words):
        seq = words_list[-seq_length:]
        seq_idx = torch.tensor([[word_to_ix.get(w, 0) for w in seq]])
        with torch.no_grad():
            prediction = model(seq_idx)
        next_word = ix_to_word[torch.argmax(prediction).item()]
        words_list.append(next_word)

    return " ".join(words_list)

print("\n✅ TEXT GENERATION MODEL READY")
prompt = input("Enter topic/prompt: ")
num_words = int(input("Enter number of words to generate: "))

output = generate_text(prompt, num_words)

print("\n📌 GENERATED TEXT:\n")
print(output)

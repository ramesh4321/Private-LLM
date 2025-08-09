import torch
import torch.nn as nn
import torch.optim as optim

# 1️⃣ Tiny dataset (pretend this is our "billions of words")
text = "patient has fever patient has pain"
words = list(set(text.split()))
print("1" , words)
word2idx = {w: i for i, w in enumerate(words)}
print("2",word2idx)
idx2word = {i: w for w, i in word2idx.items()}
print("3",idx2word)

# Convert words to indices
data = [word2idx[w] for w in text.split()]
print("4",data)

# Create input-output pairs for next-word prediction
X = []
y = []
for i in range(len(data)-1):
    X.append(data[i])
    y.append(data[i+1])

X = torch.tensor(X)
y = torch.tensor(y)

# 2️⃣ Simple neural network: Embedding + Linear layer
class TinyModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x

model = TinyModel(vocab_size=len(words), embed_size=5)

# Show random initial weights
print("🔹 Initial weights:", model.fc.weight.data)

# 3️⃣ Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4️⃣ Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Show trained weights
print("\n✅ Trained weights:", model.fc.weight.data)

# 5️⃣ Test prediction
test_word = "patient"
test_idx = torch.tensor([word2idx[test_word]])
pred = model(test_idx).argmax(dim=1).item()
print(f"\nPrediction after '{test_word}': {idx2word[pred]}")

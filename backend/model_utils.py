import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# Your model definition again
class CandidateGenerationModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim=128):  # ⬅️ Change 64 ➔ 128 here
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 256)  # 128*2=256
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, user_ids, book_ids):
        user_emb = self.user_embedding(user_ids)
        book_emb = self.book_embedding(book_ids)
        x = torch.cat([user_emb, book_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output(x))

def load_model(path, num_users, num_books):
    model = CandidateGenerationModel(num_users, num_books)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    return model

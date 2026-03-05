import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import asyncio
import numpy as np
import json
import os

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from torch.utils.data import TensorDataset, DataLoader

app = FastAPI()

# serve frontend over HTTP so that fetch requests originate from the same host
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    # return the HTML file directly
    return FileResponse("frontened.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = "experiment_history.json"

# ================= MODEL =================

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.projection = nn.Linear(128, 64)

    def forward(self, x):
        h = self.backbone(x)
        return self.projection(h)

def contrastive_loss(z1, z2, temp=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim, labels)

def corrupt(x, rate=0.4):
    """Multiple corruption strategies like real SCARF"""
    batch_size = x.size(0)
    corrupted = x.clone()
    
    # Strategy 1: Feature masking (25%)
    mask_features = torch.rand_like(x) < (rate * 0.25)
    corrupted = torch.where(mask_features, torch.zeros_like(x), corrupted)
    
    # Strategy 2: Gaussian noise (25%)
    noise_mask = torch.rand_like(x) < (rate * 0.25)
    noise = torch.randn_like(x) * 0.1
    corrupted = torch.where(noise_mask, corrupted + noise, corrupted)
    
    # Strategy 3: Feature swapping (25%)
    swap_mask = torch.rand_like(x) < (rate * 0.25)
    shuffled = x[torch.randperm(batch_size)]
    corrupted = torch.where(swap_mask, shuffled, corrupted)
    
    # Strategy 4: Sample mixing (25%)
    mix_mask = torch.rand_like(x) < (rate * 0.25)
    mixed = 0.5 * x + 0.5 * x[torch.randperm(batch_size)]
    corrupted = torch.where(mix_mask, mixed, corrupted)
    
    return corrupted

# ================= TRAIN =================

@app.post("/train/")
async def train(file: UploadFile):
    print(f"[train] received file: {file.filename}")
    try:
        df = pd.read_csv(file.file)
        df = df.select_dtypes(include=["number"])
    except Exception as e:
        print(f"[train] error reading CSV: {e}")
        raise

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Convert labels to 0-indexed
    y = y - y.min()

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=min(64, len(X_train)//4), shuffle=True)

    async def stream():
        try:
            model = Encoder(X_train.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            # -------- Pretraining with Batches --------
            for epoch in range(50):
                epoch_loss = 0
                for batch_X, _ in train_loader:
                    # Create two different corrupted views
                    x_corr1 = corrupt(batch_X)
                    x_corr2 = corrupt(batch_X)
                    
                    z1 = model(x_corr1)
                    z2 = model(x_corr2)
                    loss = contrastive_loss(z1, z2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                scheduler.step(avg_loss)

                print(f"[train] pretrain epoch {epoch} loss={avg_loss}")
                yield f"data: LOSS:{avg_loss}\n\n"
                await asyncio.sleep(0.03)

            # -------- Fine-tuning with Batches --------
            classifier = nn.Linear(128, len(np.unique(y_train)))
            optimizer = torch.optim.Adam(
                list(model.backbone.parameters()) + list(classifier.parameters()),
                lr=1e-3
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(50):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    features = model.backbone(batch_X)
                    preds = classifier(features)
                    loss = loss_fn(preds, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                scheduler.step(avg_loss)

                print(f"[train] finetune epoch {epoch} loss={avg_loss}")
                yield f"data: LOSS:{avg_loss}\n\n"
                await asyncio.sleep(0.03)

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_features = model.backbone(X_test_tensor)
                test_preds = torch.argmax(classifier(test_features), dim=1)
                scarf_acc = accuracy_score(y_test, test_preds.numpy())
            
            print(f"[train] finished training, scarf_acc={scarf_acc}")

            # Save model
            torch.save({
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'scaler': scaler
            }, 'scarf_model.pth')

            baseline = LogisticRegression(max_iter=1000)
            baseline.fit(X_train, y_train)
            baseline_acc = baseline.score(X_test, y_test)

            # -------- SAVE HISTORY --------
            try:
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r") as f:
                        history = json.load(f)
                else:
                    history = []
            except:
                history = []

            history.append({
                "dataset": file.filename,
                "scarf_accuracy": float(scarf_acc),
                "baseline_accuracy": float(baseline_acc)
            })

            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=4)

            yield f"data: DONE:{scarf_acc}:{baseline_acc}:{len(X_train)}:{len(X_test)}\n\n"
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[train] unhandled exception: {e}")
            yield f"data: ERROR:{e}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


# ================= HISTORY API =================

@app.get("/history/")
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

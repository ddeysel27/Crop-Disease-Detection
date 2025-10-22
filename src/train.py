import argparse, json, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from src.models.build_model import build
from src.utils import set_seed, save_label_mapping

def get_loaders(data_root, img=224, bs=32, num_workers=2):
    tfm_train = T.Compose([
        T.RandomResizedCrop(img, scale=(0.8,1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1,0.1,0.1,0.05),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tfm_eval = T.Compose([T.Resize((img,img)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_ds = ImageFolder(Path(data_root)/"train", transform=tfm_train)
    val_ds   = ImageFolder(Path(data_root)/"val", transform=tfm_eval)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, train_dl, val_dl

def train(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, train_dl, val_dl = get_loaders(args.data, args.img, args.bs, args.workers)
    model = build(num_classes=len(train_ds.classes), name=args.model, pretrained=True).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt  = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sch  = CosineAnnealingLR(opt, T_max=args.epochs)
    best_f1, best_path = -1.0, Path("experiments/runs/best_model.pth")
    (Path("experiments/runs")).mkdir(parents=True, exist_ok=True)
    # save label mapping
    save_label_mapping(train_ds.class_to_idx, "experiments/runs/label_mapping.json")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))

        # Validate
        model.eval()
        ys, yh = [], []
        with torch.inference_mode():
            for x,y in val_dl:
                x = x.to(device)
                logits = model(x)
                yh.extend(torch.argmax(logits,1).cpu().tolist())
                ys.extend(y.cpu().tolist())
        acc = accuracy_score(ys, yh)
        f1 = f1_score(ys, yh, average="macro")
        print(f"Val acc={acc:.4f} f1={f1:.4f}")
        sch.step()

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print("Saved", best_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--model", type=str, default="vit_base_patch16_224")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    train(args)

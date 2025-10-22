import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(weights, labels_json, data_root="data/processed/test", arch="vit_base_patch16_224", img=224):
    import timm
    idx2label = json.loads(open(labels_json).read())
    model = timm.create_model(arch, pretrained=False, num_classes=len(idx2label))
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state); model.eval()
    tfm = T.Compose([T.Resize((img,img)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = ImageFolder(data_root, transform=tfm)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    ys, yh = [], []
    with torch.inference_mode():
        for x,y in dl:
            logits = model(x)
            yh += torch.argmax(logits,1).tolist()
            ys += y.tolist()
    print(classification_report(ys, yh, target_names=ds.classes))
    cm = confusion_matrix(ys, yh)
    plt.figure()
    sns.heatmap(cm, annot=False)
    plt.title("Confusion Matrix")
    plt.savefig("experiments/runs/confusion_matrix.png", dpi=200)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/runs/best_model.pth")
    ap.add_argument("--labels", default="experiments/runs/label_mapping.json")
    ap.add_argument("--data_root", default="data/processed/test")
    ap.add_argument("--arch", default="vit_base_patch16_224")
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()
    evaluate(args.weights, args.labels, args.data_root, args.arch, args.img)

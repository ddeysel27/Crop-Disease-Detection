import torch, json
from PIL import Image
import torchvision.transforms as T

IM_SIZE = 224
_tfms = T.Compose([
    T.Resize((IM_SIZE,IM_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_model(weights, labels_json, arch="vit_base_patch16_224", device="cpu"):
    import timm
    idx2label = json.loads(open(labels_json).read())
    model = timm.create_model(arch, pretrained=False, num_classes=len(idx2label))
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, {int(k):v for k,v in idx2label.items()}

@torch.inference_mode()
def predict_topk(model, img, idx2label, k=3, device="cpu"):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    x = _tfms(img).unsqueeze(0).to(device)
    probs = model(x).softmax(1)[0]
    topk = probs.topk(k)
    scores = topk.values.cpu().tolist()
    labels = [idx2label[i] for i in topk.indices.cpu().tolist()]
    return scores, labels

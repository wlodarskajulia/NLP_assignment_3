import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(model, loader, device, model_name: str, verbose: bool = True):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    cm = confusion_matrix(all_labels, all_predictions)

    if verbose:
        print(f"{model_name} Accuracy:", acc)
        print(f"{model_name} Macro-F1:", macro_f1)
        print(f"{model_name} Confusion Matrix:\n", cm)

    return acc, macro_f1, cm, all_predictions
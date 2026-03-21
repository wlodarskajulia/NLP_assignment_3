import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_transformer(model, loader, criterion, device, model_name: str = "", verbose: bool = True):
    """
    Evaluates a HuggingFace sequence-classification model.
 
    Expects DataLoader batches as dicts with keys:
        'input_ids', 'attention_mask', 'label'
 
    Returns:
        avg_loss (float), acc (float), macro_f1 (float),
        all_predictions (list), all_labels (list)
    """
    model.eval()
    total_loss   = 0.0
    all_predictions = []
    all_labels   = []
 
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
 
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            total_loss += loss.item()
 
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())
 
    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    cm       = confusion_matrix(all_labels, all_predictions)
 
    if verbose:
        prefix = f"[{model_name}] " if model_name else ""
        print(f"{prefix}Loss     : {avg_loss:.4f}")
        print(f"{prefix}Accuracy : {acc:.4f}")
        print(f"{prefix}Macro-F1 : {macro_f1:.4f}")
        print(f"{prefix}Confusion Matrix:\n{cm}")
 
    return avg_loss, acc, macro_f1, all_predictions, all_labels
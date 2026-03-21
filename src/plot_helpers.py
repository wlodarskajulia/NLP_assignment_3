import matplotlib.pyplot as plt

def plot_learning_curves(history, title="Transformer Learning Curves"):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].plot(epochs_ran, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs_ran, history["dev_loss"],   label="Dev Loss",   marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy Loss")
    axes[0].legend()
 
    axes[1].plot(epochs_ran, history["train_f1"], label="Train Macro-F1", marker="o")
    axes[1].plot(epochs_ran, history["dev_f1"],   label="Dev Macro-F1",   marker="o")
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
 
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, title):
    # Map integers to class names
    label_map = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech"
    }
    class_names = list(label_map.values())
    _, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()
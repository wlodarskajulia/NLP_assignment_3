import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, dev_f1s, title):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, dev_f1s, label="Dev Macro-F1")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
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
import matplotlib.pyplot as plt
import numpy as np

def draw(train, test, n_epochs, title):
    epochs = np.arange(1, n_epochs + 1)
    output_file = f"./eval/{title}.png"
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, test, label="test")
    plt.title(f"{title}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(output_file)
    plt.clf()
    plt.close()

def draw_all(body):
    draw(body.history["train_loss"], body.history["test_loss"], body.n_epochs, "loss")
    draw(body.history["train_accuracy"], body.history["test_accuracy"], body.n_epochs, "accuracy")

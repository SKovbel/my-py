import matplotlib.pyplot as plt

class FitChart:
    def __init__(self):
        None

    def chart(self, model, result):
        history = result.history

        plt.figure(num="Taining info")

        # test
        acc = history['binary_accuracy'] if 'binary_accuracy' in history else history['accuracy']
        val_acc = history['val_binary_accuracy'] if 'val_binary_accuracy' in history else history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.show()

    def print(self, model, result):
        for name, value in zip(model.metrics_names, result):
            print("%s: %.3f" % (name, value))

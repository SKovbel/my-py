import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', name='conv_layer_1')
        self.conv2 = Conv2D(64, (3, 3), activation='relu', name='conv_layer_2')
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(128, activation='relu', name='dense_layer_1')
        self.dense2 = Dense(10, activation='softmax', name='output_layer')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

model = MyModel()
model.build(input_shape=(None, 28, 28, 1))

model.summary()

plot_file = 'model_plot.png'
plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)
display(Image(filename=plot_file))

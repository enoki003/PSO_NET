from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes=10,use_softmax=False):
    model = keras.Sequential(name = "sub_nn_model")
    model.add(layers.Dense(512, input_shape=(784,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes))
    optimizer = keras.optimizers.Adam()

    if use_softmax:
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', 
                      optimizer=optimizer,
                      metrics=['accuracy']
        )   
        
    else:
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=optimizer,
                      metrics=['accuracy']
        )
        

    return model
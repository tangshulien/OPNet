from tensorflow import keras
from tensorflow.keras import layers


def UNet(x0, num_classes):
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x0)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64]:

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)          
        x = layers.BatchNormalization()(x)      
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)     
        x = layers.BatchNormalization()(x)        

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        
        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )       
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        
    ### [Second half of the network: upsampling inputs] ###        
    for filters in [64]:
        
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        residual = layers.Conv2D(filters, 1, padding="same")(residual)                
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
          
    # Add a per-pixel classification layer (UNet final layer)
    x = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    # Add layers for PN
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same",name='81conv2D')(x)
    x = layers.BatchNormalization(name='66')(x)
    x = layers.Activation("relu",name='67')(x)
    x = layers.Flatten(name='84')(x)

    return x

# PN will be used in future
def PN(x):
    x1 = layers.Dense(64, activation='relu', name="74")(x)
    x2 = layers.Dense(64, activation='relu', name="75")(x)
    x3 = layers.Dense(64, activation='relu', name="76")(x)
    x4 = layers.Dense(64, activation='relu', name="77")(x) 
    x5 = layers.Dense(64, activation='relu', name="78")(x)
    x6 = layers.Dense(64, activation='relu', name="79")(x)
    x7 = layers.Dense(64, activation='relu', name="80")(x)
    x8 = layers.Dense(64, activation='relu', name="81")(x)
    #x9 = layers.Dense(64, activation='relu', name="98")(x)
    #x10 = layers.Dense(64, activation='relu', name="99")(x)
    #x11 = layers.Dense(64, activation='relu', name="100")(x) 
    
    out1 = layers.Dense(386, name="1path")(x1)
    out2 = layers.Dense(200, name="5long_x")(x2)
    out3 = layers.Dense(200, name="7long_a")(x3)    
    out4 = layers.Dense(58, name="l4ead")(x4)   
    out5 = layers.Dense(386, name="2left_lane")(x5)    
    out6 = layers.Dense(200, name="6long_v")(x6)   
    out7 = layers.Dense(8, name="8desire_state")(x7)
    out8 = layers.Dense(386, name="3right_lane")(x8)
    #out9 = layers.Dense(12, name="11pose")(x9)    
    #out10 = layers.Dense(4, name="9meta")(x10)     
    #out11 = layers.Dense(32, name="10desire_pred")(x11)      
    out12 = (x)
    outputs = layers.Concatenate(axis=-1)([out1, out2, out3, out4, out5, out6, out7, out8, out12])
    #outputs = layers.Concatenate(axis=-1)([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11])
    return outputs

def get_model(img_shape, num_classes):
    inputs = keras.Input(shape=img_shape)
    x0 = layers.Permute((2, 3, 1))(inputs)
    x = UNet(x0, num_classes)
    outputs = PN(x)

    model = keras.Model(inputs, outputs)

    return model

if __name__=="__main__":
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    img_shape = (12, 128, 256)
    num_classes = 3
    model = get_model(img_shape, num_classes)
    model.summary()

    keras.utils.plot_model(model, "./saved_model/T8_modelB3.png")
    keras.utils.plot_model(model, "./saved_model/T8_modelB3.png", show_shapes=True)  




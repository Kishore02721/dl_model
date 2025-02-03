import tensorflow as tf

def encoder_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    return x

def decoder_block(inputs, skip1, skip2, num_filters, merge_mode='concat'):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)

    # Use UpSampling2D for resizing, as it's compatible with Keras tensors
    skip1 = tf.keras.layers.UpSampling2D(size=(2, 2))(skip1)
    skip2 = tf.keras.layers.UpSampling2D(size=(2, 2))(skip2)

    if merge_mode == 'concat':
        merged_skip = tf.keras.layers.Concatenate()([skip1, skip2])
    elif merge_mode == 'diff':
        merged_skip = tf.keras.layers.Subtract()([skip1, skip2])
    else:
        raise ValueError("merge_mode should be 'concat', 'diff', or 'multiply'")
    
    x = tf.keras.layers.Concatenate()([x, merged_skip])
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def unet_with_two_encoders(input_shape=(256, 256, 1), num_classes=1):
    original_image = tf.keras.layers.Input(input_shape, name="original_image")
    inspected_image = tf.keras.layers.Input(input_shape, name="inspected_image")
    
    s1 = encoder_block(original_image, 64)
    s2 = encoder_block(s1, 128)
    s3 = encoder_block(s2, 256)
    s4 = encoder_block(s3, 512)
    
    t1 = encoder_block(inspected_image, 64)
    t2 = encoder_block(t1, 128)
    t3 = encoder_block(t2, 256)
    t4 = encoder_block(t3, 512)
    
    # Ensure concatenation is done inside a layer, not a TensorFlow function
    concat_input = tf.keras.layers.Concatenate(axis=-1)([s4, t4])  # Concatenate encoders' outputs
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same')(concat_input)  # Apply Conv2D
    b1 = tf.keras.layers.Activation('relu')(b1)
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same')(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)
    
    d1 = decoder_block(b1, s4, t4, 512, merge_mode='concat')
    d2 = decoder_block(d1, s3, t3, 256, merge_mode='concat')
    d3 = decoder_block(d2, s2, t2, 128, merge_mode='concat')
    d4 = decoder_block(d3, s1, t1, 64, merge_mode='concat')

    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)
    
    model = tf.keras.models.Model(inputs=[original_image, inspected_image], outputs=outputs, name="U-Net_with_Two_Encoders")
    
    return model

if __name__ == "__main__":
    model = unet_with_two_encoders(input_shape=(256, 256, 1), num_classes=1)
    model.summary()


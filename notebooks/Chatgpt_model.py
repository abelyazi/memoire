import tensorflow as tf

def initial_convolution(input_tensor):
    # Initial convolution and max pooling
    conv_layer = tf.keras.layers.Conv1D(filters=12, kernel_size=7, strides=2, activation='relu', padding='same')(input_tensor)
    maxpool_layer = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(conv_layer)
    
    return maxpool_layer

def composite_layer(input_tensor):
    # Composite layer within clique block
    expansion_ratio = 6
    intermediate_channels = input_tensor.shape[-1] * expansion_ratio
    
    # Point-wise convolution for channel expansion
    input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
    expanded_features = tf.keras.layers.Conv1D(filters=intermediate_channels, kernel_size=1, padding='same')(input_tensor)
    
    # Depth-wise convolution
    expanded_features = tf.keras.layers.BatchNormalization()(expanded_features)
    expanded_features = tf.keras.layers.ReLU()(expanded_features)
    depthwise_conv = tf.keras.layers.DepthwiseConv1D(kernel_size=3, strides=1, padding='same')(expanded_features)
    
    # Point-wise convolution for channel reduction
    depthwise_conv = tf.keras.layers.BatchNormalization()(depthwise_conv)
    depthwise_conv = tf.keras.layers.ReLU()(depthwise_conv)
    bottleneck_features = tf.keras.layers.Conv1D(filters=12, kernel_size=1, padding='same')(depthwise_conv)
    
    return bottleneck_features

def clique_block(input_tensor):
    
    # Stage 1
    feat_0 = input_tensor
    feat_1 = composite_layer(feat_0)
    feat_0_1 = tf.keras.layers.concatenate([feat_0, feat_1], axis=-1)
    feat_2 = composite_layer(feat_0_1)
    feat_0_1_2 = tf.keras.layers.concatenate([feat_0, feat_1, feat_2], axis=-1)
    feat_3 = composite_layer(feat_0_1_2)
    feat_0_1_2_3 = tf.keras.layers.concatenate([feat_0, feat_1, feat_2, feat_3], axis=-1)
    feat_4 = composite_layer(feat_0_1_2_3)
    feat_0_1_2_3_4 = tf.keras.layers.concatenate([feat_0, feat_1, feat_2, feat_3, feat_4], axis=-1)
    feat_5 = composite_layer(feat_0_1_2_3_4)
    
    # Stage 2
    feat_3_4_5 = tf.keras.layers.concatenate([feat_3, feat_4, feat_5], axis=-1)
    feat_6 = composite_layer(feat_3_4_5)
    feat_4_5_6 = tf.keras.layers.concatenate([feat_4, feat_5, feat_6], axis=-1)
    feat_7 = composite_layer(feat_4_5_6)
    feat_5_6_7 = tf.keras.layers.concatenate([feat_5, feat_6, feat_7], axis=-1)
    feat_8 = composite_layer(feat_5_6_7)
    feat_6_7_8 = tf.keras.layers.concatenate([feat_6, feat_7, feat_8], axis=-1)
    feat_9 = composite_layer(feat_6_7_8)
    feat_7_8_9 = tf.keras.layers.concatenate([feat_7, feat_8, feat_9], axis=-1)
    feat_10 = composite_layer(feat_7_8_9)
    
    return feat_10

def transition_block(input_tensor):
    # Transition block without dimension reduction
    bn = tf.keras.layers.BatchNormalization()(input_tensor)
    relu = tf.keras.layers.ReLU()(bn)
    pconv = tf.keras.layers.Conv1D(filters=12, kernel_size=1, padding='same')(relu)
    
    glob_pool = tf.keras.layers.GlobalAveragePooling1D()(pconv)
    dense = tf.keras.layers.Dense(12)(glob_pool)
    relu = tf.keras.layers.ReLU()(dense)
    dense = tf.keras.layers.Dense(12)(relu)
    sigmoid = tf.keras.activations.sigmoid(dense)
    
    scale = tf.keras.layers.Multiply()([sigmoid, pconv])
    avg_pool = tf.keras.layers.AveragePooling1D(pool_size=2)(scale)
        
    return avg_pool


def create_small_model(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    # Initial convolution and max pooling
    maxpool_layer = initial_convolution(inputs)
    
    # Clique blocks
    clique_block1 = clique_block(maxpool_layer)
    transition_block1 = transition_block(clique_block1)
    clique_block2 = clique_block(transition_block1)
    transition_block2 = transition_block(clique_block2)
    clique_block3 = clique_block(transition_block2)
  
    
    #maxpool_clique_block1 = tf.keras.layers.concatenate([maxpool_layer, clique_block1], axis=-1)

    merge_1 = tf.keras.layers.concatenate([maxpool_layer, clique_block1], axis=-1)
    merge_2 = tf.keras.layers.concatenate([transition_block1, clique_block2], axis=-1)
    merge_3 = tf.keras.layers.concatenate([transition_block2, clique_block3], axis=-1)
    
    # Squeezed multi-scale representation
    pooled_1 = tf.keras.layers.GlobalAveragePooling1D()(merge_1)
    pooled_2 = tf.keras.layers.GlobalAveragePooling1D()(merge_2)
    pooled_3 = tf.keras.layers.GlobalAveragePooling1D()(merge_3)
    
    merge_pool_2_3 = tf.keras.layers.concatenate([pooled_2, pooled_3], axis=-1)
    merge_pool_1_23 = tf.keras.layers.concatenate([pooled_1, merge_pool_2_3], axis=-1)
    
    # Merge squeezed features
    #merged_features = tf.keras.layers.concatenate([pooled_features1, pooled_features2, pooled_features3], axis=-1)
    
    # Classification layer
    x = tf.keras.layers.Dense(12)(merge_pool_1_23)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution and max pooling
    maxpool_layer = initial_convolution(inputs)
    
    # Clique blocks
    clique_block1 = clique_block(maxpool_layer)
    transition_block1 = transition_block(clique_block1)
    
    clique_block2 = clique_block(transition_block1)
    transition_block2 = transition_block(clique_block2)
    
    clique_block3 = clique_block(transition_block2)
    
    # Squeezed multi-scale representation
    pooled_features1 = tf.keras.layers.GlobalAveragePooling1D()(clique_block1)
    pooled_features2 = tf.keras.layers.GlobalAveragePooling1D()(clique_block2)
    pooled_features3 = tf.keras.layers.GlobalAveragePooling1D()(clique_block3)
    
    # Merge squeezed features
    merged_features = tf.keras.layers.concatenate([pooled_features1, pooled_features2, pooled_features3], axis=-1)
    
    # Classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(merged_features)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
#input_shape = (input_length, num_channels)  # Replace with the actual input length and number of channels
#num_classes = 10  # Replace with the actual number of classes
#model = create_model(input_shape, num_classes)

# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy

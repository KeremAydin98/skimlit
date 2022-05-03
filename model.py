import tensorflow as tf


def create_model(char_embedding, char_vectorization, token_embeddings_layer):
    # 1. Token embeddings
    token_inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name="token_input")
    x = token_embeddings_layer(token_inputs)
    token_outputs = tf.keras.layers.Dense(128, activation="relu")(x)

    token_model = tf.keras.models.Model(token_inputs, token_outputs)

    # 2. Char vectorization
    char_inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="char_input")
    x = char_vectorization(char_inputs)
    x = char_embedding(x)
    char_outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24))(x)

    char_model = tf.keras.models.Model(char_inputs, char_outputs)

    # 3. Line number feature
    line_number_input = tf.keras.layers.Input(shape=(15,), dtype=tf.float64, name="line_number_input")
    line_number_output = tf.keras.layers.Dense(128, activation="relu")(line_number_input)

    line_number_model = tf.keras.models.Model(line_number_input, line_number_output)

    # 4. Total lines feature
    total_lines_input = tf.keras.layers.Input(shape=(20,), dtype=tf.float64, name="total_lines_input")
    total_lines_output = tf.keras.layers.Dense(128, activation="relu")(total_lines_input)

    total_lines_model = tf.keras.models.Model(total_lines_input, total_lines_output)

    # 5. Concatenate token and char models
    concat_embeddings = tf.keras.layers.Concatenate()([token_model.output,
                                                       char_model.output])

    z = tf.keras.layers.Dense(256, activation="relu")(concat_embeddings)
    z = tf.keras.layers.Dropout(0.5)(z)

    # 6. Tribrid model
    tribrid_output = tf.keras.layers.Concatenate()([line_number_model.output,
                                                    total_lines_model.output,
                                                    z])

    # 7. Output layer
    z = tf.keras.layers.Dropout(0.5)(tribrid_output)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(z)

    # 8. Combine all the models
    model = tf.keras.models.Model(inputs=[token_model.input,
                                          char_model.input,
                                          line_number_model.input,
                                          total_lines_model.input],
                                  outputs=[outputs])

    # Compile the model
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["accuracy"])

    return model

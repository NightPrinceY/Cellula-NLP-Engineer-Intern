def train_model(train_csv, eval_csv, tokenizer_path, model_path, max_words=10000, max_len=150, num_classes=9):
    import os
    import pandas as pd
    import pickle
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping

    # Load data
    train_df = pd.read_csv(train_csv)
    eval_df = pd.read_csv(eval_csv)
    def build_model(vocab_size, max_len, num_classes):
        model = tf.keras.Sequential([
             tf.keras.layers.Embedding(vocab_size, 128, input_length=max_len),
             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
             tf.keras.layers.GlobalMaxPooling1D(),
             tf.keras.layers.Dropout(0.3),
             tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dropout(0.3),
             tf.keras.layers.Dense(num_classes, activation='softmax')
           ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Prepare text and labels
    X_train = tokenizer.texts_to_sequences(train_df['query'] + ' ' + train_df['image descriptions'])
    X_eval = tokenizer.texts_to_sequences(eval_df['query'] + ' ' + eval_df['image descriptions'])
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_eval = pad_sequences(X_eval, maxlen=max_len, padding='post', truncating='post')
    y_train = train_df['Toxic Category Encoded'].values
    y_eval = eval_df['Toxic Category Encoded'].values

    # Build and train model
    model = build_model(max_words, max_len, num_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_eval, y_eval),
        callbacks=[early_stop]
    )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save model
    model.save(model_path)


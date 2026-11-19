import os
import numpy as np
import tensorflow as tf

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
SEQ_LENGTH = 100        # Length of text sequence to look at
BATCH_SIZE = 64         # Number of sequences per training batch
BUFFER_SIZE = 10000     # For dataset shuffling
EPOCHS = 10             # Number of training loops
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================================
# 2. DATA PREPARATION PIPELINE
# ==========================================
def load_and_prep_data():
    print("Downloading dataset...")
    path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt',
        cache_dir='.', cache_subdir='data'
    )

    with open(path_to_file, 'rb') as f:
        text = f.read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    # Create text <-> integer lookup layers
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )

    # Convert entire text to integers
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    # Batch sequences
    sequences = ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    # Split sequences into input (x) and target (y)
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Shuffle and pack into training batches
    dataset = (dataset
               .shuffle(BUFFER_SIZE)
               .batch(BATCH_SIZE, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))

    return dataset, ids_from_chars, chars_from_ids, len(ids_from_chars.get_vocabulary())

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class CharModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            x, states = self.gru(x, training=training)
        else:
            x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        return x

# ==========================================
# 4. TEXT GENERATION (WITH TEMPERATURE)
# ==========================================
class OneStepGenerate(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Mask to prevent [UNK] token generation
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())]
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run model and get logits for the last character
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]

        # Apply temperature and mask out unknown tokens
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample from the adjusted distribution
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states

# ==========================================
# 5. MAIN EXECUTION (TRAIN & GENERATE)
# ==========================================
if __name__ == "__main__":
    # 1. Prepare data
    dataset, ids_from_chars, chars_from_ids, vocab_size = load_and_prep_data()

    # 2. Build model
    model = CharModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    # 3. Setup Checkpoints (Save model weights during training)
    checkpoint_prefix = os.path.join("models", "ckpt_{epoch}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    # 4. Train the model
    print("Starting training...")
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # 5. Generate Text
    print("\n--- Generating Text ---")
    one_step_model = OneStepGenerate(model, chars_from_ids, ids_from_chars, temperature=0.7)

    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]

    # Generate 500 characters
    for _ in range(500):
        next_char, states = one_step_model.generate_step(next_char, states=states)
        result.append(next_char)

    generated_text = tf.strings.join(result)[0].numpy().decode('utf-8')
    print(generated_text)

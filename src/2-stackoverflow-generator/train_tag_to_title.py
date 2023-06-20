import os
import pathlib

import tensorflow as tf
from tensorflow import keras
import keras_nlp

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
keras.mixed_precision.set_global_policy("mixed_float16")

# Data
BATCH_SIZE = 32
SEQ_LEN = 256
MIN_TRAINING_SEQ_LEN = 20

# Model
EMBED_DIM = 65
FEED_FORWARD_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # Limits parameters in model.

# Training
EPOCHS = 6

# Inference
NUM_TOKENS_TO_GENERATE = 80


def preprocess(pre_inputs):
    pre_outputs = tokenizer(pre_inputs)
    features = start_packer(pre_outputs)
    return features, pre_outputs


def generate_filenames(path_str):
    path = pathlib.Path(path_str)

    for file_name in os.listdir(path):
        yield path / file_name


if __name__ == "__main__":
    train_files = [fn for fn in generate_filenames("processed/titles/train")]
    validation_files = [fn for fn in generate_filenames("processed/titles/validation")]

    raw_train_ds = (
        tf.data.TextLineDataset(train_files)
        .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
        .batch(BATCH_SIZE)
        .shuffle(buffer_size=256)
    )

    raw_val_ds = (
        tf.data.TextLineDataset(validation_files)
        .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
        .batch(BATCH_SIZE)
    )

    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        raw_train_ds,
        vocabulary_size=VOCAB_SIZE,
        lowercase=True,
        reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
    )

    for v in vocab:
        print(v)

    # tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    #     vocabulary=vocab,
    #     sequence_length=SEQ_LEN,
    #     lowercase=True,
    # )
    #
    # start_packer = keras_nlp.layers.StartEndPacker(
    #     sequence_length=SEQ_LEN,
    #     start_value=tokenizer.token_to_id("[BOS]"),
    # )
    #
    # # Tokenize and split into train and label sequences.
    # train_ds = raw_train_ds.map(
    #     preprocess, num_parallel_calls=tf.data.AUTOTUNE
    # ).prefetch(tf.data.AUTOTUNE)
    # val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    #     tf.data.AUTOTUNE
    # )
    #
    # inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
    # # Embedding.
    # embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    #     vocabulary_size=VOCAB_SIZE,
    #     sequence_length=SEQ_LEN,
    #     embedding_dim=EMBED_DIM,
    #     mask_zero=True,
    # )
    # x = embedding_layer(inputs)
    # # Transformer decoders.
    # for _ in range(NUM_LAYERS):
    #     decoder_layer = keras_nlp.layers.TransformerDecoder(
    #         num_heads=NUM_HEADS,
    #         intermediate_dim=FEED_FORWARD_DIM,
    #     )
    #     x = decoder_layer(x)  # Giving one argument only skips cross-attention.
    # # Output.
    # outputs = keras.layers.Dense(VOCAB_SIZE)(x)
    # model = keras.Model(
    #     inputs=inputs,
    #     outputs=outputs,
    # )
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True,
    # )
    # perplexity = keras_nlp.metrics.Perplexity(
    #     from_logits=True,
    #     mask_token_id=0,
    # )
    # model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])
    #
    # model.summary()
    #
    # checkpoint_path = "checkpoints/tag-to-title/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    # # Create a callback that saves the model's weights every 5 epochs
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     verbose=1,
    #     save_weights_only=True,
    #     save_freq=1,
    # )
    #
    # # Save the weights using the `checkpoint_path` format
    # model.save_weights(checkpoint_path.format(epoch=0))
    #
    # model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     verbose=2,
    #     epochs=EPOCHS,
    #     callbacks=[cp_callback],
    # )
    #
    # model.save("test_model.h5")
    #
    #
    # def next(prompt, cache, index):
    #     logits = model(prompt)[:, index - 1, :]
    #     # Ignore hidden states for now; only needed for contrastive search.
    #     hidden_states = None
    #     return logits, hidden_states, cache
    #
    #
    # while True:
    #     user_input = input("Enter prompt: ")
    #     prompt_tokens = start_packer(tokenizer([user_input]))
    #     print(prompt_tokens)
    #
    #     sampler = keras_nlp.samplers.GreedySampler()
    #     output_tokens = sampler(
    #         next=next,
    #         prompt=prompt_tokens,
    #         index=1,  # Start sampling immediately after the [BOS] token.
    #     )
    #     txt = tokenizer.detokenize(output_tokens)
    #     print(f"Greedy search generated text: \n{txt}\n")

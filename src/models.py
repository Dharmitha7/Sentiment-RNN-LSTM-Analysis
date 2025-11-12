# src/models.py
"""
Model builders for sentiment classification with RNN variants.

Supported architectures:
- "rnn"    : SimpleRNN
- "lstm"   : LSTM
- "bilstm" : Bidirectional LSTM

Notes:
- Use build_model(...) to get a tf.keras.Sequential model.
- Compile in train.py so you can swap optimizers, add gradient clipping, etc.
"""

from typing import Literal
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
)

ArchType = Literal["rnn", "lstm", "bilstm"]
ActType = Literal["tanh", "relu", "sigmoid"]  

def _make_recurrent_layer(
    arch: ArchType,
    units: int,
    activation: str,
    return_sequences: bool,
    dropout: float,
    recurrent_dropout: float = 0.0,
):
    """
    Construct a single recurrent layer (possibly bidirectional).
    For "bilstm", wraps an LSTM with Bidirectional.
    """
    if arch == "rnn":
        layer = SimpleRNN(
            units,
            activation=activation,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        return layer

    if arch == "lstm":
        layer = LSTM(
            units,
            activation=activation,        # cell activation (default tanh)
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        return layer

    if arch == "bilstm":
        base = LSTM(
            units,
            activation=activation,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        return Bidirectional(base)

    raise ValueError(f"Unknown arch '{arch}'. Use 'rnn', 'lstm', or 'bilstm'.")


def build_model(
    arch: ArchType = "rnn",
    vocab_size: int = 10000,
    seq_len: int = 50,
    emb_dim: int = 100,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.3,
    activation: ActType = "tanh",
    recurrent_dropout: float = 0.0,
) -> tf.keras.Model:
    """
    Build a Sequential model according to the assignment specs.

    Parameters
    ----------
    arch : "rnn" | "lstm" | "bilstm"
        Recurrent architecture to use.
    vocab_size : int
        Size of the tokenizer vocabulary (pass len(word_index)+1 capped by max_words).
    seq_len : int
        Input sequence length (25, 50, or 100 per your experiments).
    emb_dim : int
        Embedding dimension (assignment suggests 100).
    hidden : int
        Hidden size for recurrent layers (assignment suggests 64).
    layers : int
        Number of recurrent layers (assignment: 2).
    dropout : float
        Dropout rate after recurrent layers (0.3-0.5).
    activation : "tanh" | "relu" | "sigmoid"
        Activation inside the recurrent cell (SimpleRNN/LSTM).
        (LSTM also uses internal sigmoid gates; this sets the cell/output activation.)
    recurrent_dropout : float
        Dropout applied to recurrent state (can slow down on CPU; default 0.0).

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model. Compile in train.py.
    """
    if arch not in {"rnn", "lstm", "bilstm"}:
        raise ValueError("arch must be one of: 'rnn', 'lstm', 'bilstm'")
    if activation not in {"tanh", "relu", "sigmoid"}:
        raise ValueError("activation must be one of: 'tanh', 'relu', 'sigmoid'")

    model = Sequential(name=f"{arch}_sentiment")

    # Embedding layer (input: integer token ids)
    model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=seq_len))

    # Stack recurrent layers:
    # - All but the last recurrent layer use return_sequences=True
    # - Last recurrent layer outputs a final vector
    for i in range(max(layers - 1, 0)):
        model.add(
            _make_recurrent_layer(
                arch=arch,
                units=hidden,
                activation=activation,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
        model.add(Dropout(dropout))

    # Final recurrent layer (no return_sequences)
    model.add(
        _make_recurrent_layer(
            arch=arch,
            units=hidden,
            activation=activation,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
    )
    model.add(Dropout(dropout))

    # Binary classifier head
    model.add(Dense(1, activation="sigmoid"))

    return model

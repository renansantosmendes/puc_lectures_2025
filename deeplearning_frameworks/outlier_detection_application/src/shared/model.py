import torch
import torch.nn as nn

class ForecastLSTMAutoencoder(nn.Module):
    """
    LSTM Encoder with Multi-Head Attention and Decoder for time series forecasting.

    This model encodes input sequences using an LSTM, applies multi-head attention
    between a latent representation and encoder outputs, then decodes to predict
    the next timestep.

    Attributes:
        sequence_length: Length of input sequences.
        input_dimension: Number of input features.
        hidden_dimension: Size of LSTM hidden state.
        latent_dimension: Size of latent representation.
        number_of_layers: Number of LSTM layers.
    """

    def __init__(
        self,
        sequence_length: int,
        input_dimension: int,
        hidden_dimension: int,
        latent_dimension: int,
        number_of_layers: int = 2,
        dropout_rate: float = 0.2,
        number_of_heads: int = 4
    ) -> None:
        """
        Initialize the ForecastLSTMAutoencoder.

        Args:
            sequence_length: Length of input sequences.
            input_dimension: Number of input features.
            hidden_dimension: Size of LSTM hidden state.
            latent_dimension: Size of latent representation.
            number_of_layers: Number of LSTM layers.
            dropout_rate: Dropout probability.
            number_of_heads: Number of attention heads.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.latent_dimension = latent_dimension
        self.number_of_layers = number_of_layers

        self.encoder_lstm = nn.LSTM(
            input_dimension,
            hidden_dimension,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=dropout_rate if number_of_layers > 1 else 0
        )

        self.encoder_fully_connected = nn.Linear(hidden_dimension, latent_dimension)
        self.encoder_layer_norm = nn.LayerNorm(latent_dimension)
        self.dropout = nn.Dropout(dropout_rate)

        self.query_projection = nn.Linear(latent_dimension, hidden_dimension)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=hidden_dimension,
            num_heads=number_of_heads,
            batch_first=True
        )

        self.decoder_fully_connected = nn.Linear(hidden_dimension, hidden_dimension)
        self.decoder_layer_norm = nn.LayerNorm(hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, input_dimension)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_tensor: Input tensor with shape (batch, sequence_length, input_dimension).

        Returns:
            Predicted values for the next timestep with shape (batch, input_dimension).
        """
        encoder_outputs, (hidden_state, cell_state) = self.encoder_lstm(input_tensor)

        last_hidden_state = hidden_state[-1]
        latent_representation = self.encoder_fully_connected(last_hidden_state)
        latent_representation = self.encoder_layer_norm(latent_representation)
        latent_representation = self.dropout(latent_representation)

        query = self.query_projection(latent_representation).unsqueeze(1)
        attention_output, attention_weights = self.multi_head_attention(
            query,
            encoder_outputs,
            encoder_outputs
        )
        attention_output = attention_output.squeeze(1)

        decoded = self.decoder_fully_connected(attention_output)
        decoded = self.decoder_layer_norm(decoded)
        decoded = self.dropout(decoded)

        prediction = self.output_layer(decoded)
        return prediction

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequences to latent representation.

        Args:
            input_tensor: Input tensor with shape (batch, sequence_length, input_dimension).

        Returns:
            Latent representation with shape (batch, latent_dimension).
        """
        encoder_outputs, (hidden_state, cell_state) = self.encoder_lstm(input_tensor)
        last_hidden_state = hidden_state[-1]
        latent_representation = self.encoder_fully_connected(last_hidden_state)
        latent_representation = self.encoder_layer_norm(latent_representation)
        return latent_representation

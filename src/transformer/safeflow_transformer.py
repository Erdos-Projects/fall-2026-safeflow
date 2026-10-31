import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeFlowTransformer(nn.Module):
    def __init__(self, n_features, max_history=24, d_model=64, nhead=4, num_layers=2, d_ff=128, dropout=0.1):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model is not divisible by nhead")

        # Port features fo each time step into encoder representation
        self.input_projection = nn.Linear(n_features, d_model)

        self.position = nn.Parameter(torch.randn(1, max_history, d_model) * 0.02)

        # Define encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)

        # Initialize weights
        for param in self.encoder.parameters():
            if param.ndim > 1:
                nn.init.xavier_uniform_(param)

        self.output = nn.Linear(d_model, 1)

    # >>> LOCAL NOTEBOOK CHANGE: optional mask for unavailable leading history.
    def forward(self, x, padding_mask=None):
        history_length = x.shape[1]

        if history_length > self.position.shape[1]:
            raise ValueError("Max history exceeded")

        x = self.input_projection(x)
        x = x + self.position[:, :history_length]

        # Bidirectional attention is allowed because all tokens are historical
        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2]:
                raise ValueError("padding_mask must have shape (batch, history)")
            if padding_mask[:, -1].any():
                raise ValueError("The current observation must be a valid token")
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # <<< LOCAL NOTEBOOK CHANGE

        return self.output(x[:, -1]).squeeze(-1)

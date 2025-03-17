import torch.nn as nn

class ModularMLP(nn.Module):
    def __init__(
        self,
        input_dim=1536,
        hidden_dim=1024,
        num_hidden_layers=2,
        dropout=0.1,          # dropout probability
        output_dim=768        # final dimension (e.g., 768 to match BERT's hidden size)
    ):
        """
        A multi-layer perceptron with multiple hidden layers, ReLU activations,
        and dropout for regularization.
        """
        super().__init__()

        layers = []

        # 1) Input -> first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # 2) Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(int(hidden_dim/(2**(_))), int(hidden_dim/(2**(_+1)))))
            layers.append(nn.Dropout(dropout))

        # 3) Final projection layer
        self.out_proj = nn.Linear(int(hidden_dim/2**(num_hidden_layers-1)), output_dim)

        # Wrap in ModuleList so that they become part of the model
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        # Final linear layer
        x = self.out_proj(x)
        return x
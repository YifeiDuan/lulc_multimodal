import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_units, activation=nn.ReLU):
        """
        Initialize the Multi-Layer Perceptron.

        Parameters:
        - input_dim (int): Number of input features.
        - output_dim (int): Number of output units.
        - hidden_layers (int): Number of hidden layers.
        - hidden_units (int or list): Number of units in each hidden layer. If a single integer is given,
                                       all hidden layers will have the same number of units.
        - activation (nn.Module): Activation function to use. Default is ReLU.
        """
        super(MLP, self).__init__()

        # Handle case where hidden_units is a single integer or a list
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units] * hidden_layers
        elif len(hidden_units) != hidden_layers:
            raise ValueError(f"Length of hidden_units ({len(hidden_units)}) must be equal to hidden_layers ({hidden_layers})")

        # Build the layers
        layers = []
        prev_units = input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_units, hidden_units[i]))
            layers.append(activation())  # Apply activation after each layer
            prev_units = hidden_units[i]
        
        # Output layer
        layers.append(nn.Linear(prev_units, output_dim))

        # Stack layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.network(x)


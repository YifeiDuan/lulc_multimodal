import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_layers=1, hidden_units=1024, activation=nn.ReLU):
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


class LateFusionModel(nn.Module):
    def __init__(self, input_dim_img, input_dim_txt, output_dim=1, hidden_layers=1, hidden_units=1024, activation=nn.ReLU):
        """
        Initialize the Multi-Layer Perceptron.

        Parameters:
        - input_dim_img (int): Dim of input img embedding.
        - input_dim_txt (int): Dim of input txt embedding.
        - output_dim (int): Number of output units.
        - hidden_layers (int): Number of hidden layers.
        - hidden_units (int or list): Number of units in each hidden layer. If a single integer is given,
                                       all hidden layers will have the same number of units.
        - activation (nn.Module): Activation function to use. Default is ReLU.
        """
        super(LateFusionModel, self).__init__()

        # Handle case where hidden_units is a single integer or a list
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units] * hidden_layers
        elif len(hidden_units) != hidden_layers:
            raise ValueError(f"Length of hidden_units ({len(hidden_units)}) must be equal to hidden_layers ({hidden_layers})")

        # Build the layers for img_adaptor
        layers = []
        prev_units = input_dim_img
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_units, hidden_units[i]))
            layers.append(activation())  # Apply activation after each layer
            prev_units = hidden_units[i]
        self.img_adaptor = nn.Sequential(*layers)

        # Build the layers for txt_adaptor
        layers = []
        prev_units = input_dim_txt
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_units, hidden_units[i]))
            layers.append(activation())  # Apply activation after each layer
            prev_units = hidden_units[i]
        self.txt_adaptor = nn.Sequential(*layers)

        # Build the final classifier
        layers = []
        prev_units = hidden_units[-1] * 2   # To concatenate adapted img and txt embeddings
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_units, hidden_units[i]))
            layers.append(activation())  # Apply activation after each layer
            prev_units = hidden_units[i]
        layers.append(nn.Linear(prev_units, output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x1, x2):
        """
        Forward pass through the network.
        
        Parameters:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor.
        """
        x1 = self.img_adaptor(x1)
        x2 = self.txt_adaptor(x2)
        x = torch.cat((x1, x2), dim=-1)
        return self.classifier(x)
import abc

import torch
from . import bsq



def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer_3 = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=16, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_2 = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.linear_layer = torch.nn.Linear(d_latent, n_tokens)
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(x.shape) == 4:
            x = bsq.load().encode_index(x.float() / 254.5)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        B, h, w = x.shape
        embedded = self.embedding(torch.nn.functional.pad(x.reshape(B, -1)[:, :-1], (1, 0), value=0))
        # transformed_3 = self.transformer_3(embedded, src_mask=torch.nn.Transformer.generate_square_subsequent_mask(h * w).to(x.device))
        # transformed_2 = self.transformer_2(embedded, src_mask=torch.nn.Transformer.generate_square_subsequent_mask(h * w).to(x.device))
        transformed = self.transformer(embedded, src_mask=torch.nn.Transformer.generate_square_subsequent_mask(h * w).to(x.device))
        output = self.output_proj(transformed)
        return output.reshape(B, h, w, -1), {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        t = h * w
        zeroes = torch.zeros((B, t), dtype=torch.long, device=device)
        for i in range(t):
            embedded = self.embedding(torch.nn.functional.pad(zeroes[:, :i+1], (0, t - i-1), value=0))
            # transformed_3 = self.transformer_3(embedded, src_mask=torch.nn.Transformer.generate_square_subsequent_mask(t).to(device))
            # transformed_2 = self.transformer_2(embedded, src_mask=torch.nn.Transformer.generate_square_subsequent_mask(t).to(device))
            transformed = self.transformer(embedded, src_mask=torch.nn.Transformer.generate_square_subsequent_mask(t).to(device))
            zeroes[:, i] = torch.multinomial(torch.nn.functional.softmax(self.output_proj(transformed[:, i, :]), dim=-1), 1).squeeze(-1)
        return zeroes.reshape(B, h, w)

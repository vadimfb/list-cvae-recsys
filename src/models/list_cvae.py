from typing import List, Dict
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import DataLoader


class VariationalEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(VariationalEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        layers = []
        input_dim = self.input_dim
        for dim in self.hidden_dims:
            fc = nn.Linear(input_dim, dim)
            layers.append(fc)
            layers.append(nn.ReLU())
            input_dim = dim

        self.hidden_layers = nn.Sequential(*layers)
        self.last_fc = nn.Linear(input_dim, 2 * self.latent_dim)

    def forward(self, x: torch.Tensor):

        x = self.hidden_layers(x)
        x = self.last_fc(x)

        mu_z = x[:, : self.latent_dim]
        log_std_z = x[:, self.latent_dim:]

        return mu_z, log_std_z


class Decoder(nn.Module):

    def __init__(self, output_dim: int, latent_dim: int, hidden_dims: List[int]):

        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        layers = []
        input_dim = self.latent_dim
        for dim in self.hidden_dims:
            fc = nn.Linear(input_dim, dim)
            layers.append(fc)
            layers.append(nn.ReLU())
            input_dim = dim

        self.hidden_layers = nn.Sequential(*layers)
        self.last_fc = nn.Linear(input_dim, self.output_dim)

    def forward(self, z):

        z = self.hidden_layers(z)
        x_recon = self.last_fc(z)

        return x_recon


class ListCVAE(nn.Module):

    def __init__(self,
                 num_items: int,
                 slate_size: int,
                 embedding_dim: int,
                 latent_dim: int,
                 encoder_hidden_dims: List[int],
                 prior_hidden_dims: List[int],
                 decoder_hidden_dims: List[int],
                 device: str):

        super(ListCVAE, self).__init__()

        self.num_items = num_items
        self.slate_size = slate_size
        self.embedding_dim = embedding_dim
        self.device = device

        self.item_embeddings = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx=self.num_items)
        self.all_items = torch.arange(self.num_items + 1, device=self.device)

        encoded_vector_dim = self.slate_size * self.embedding_dim + self.slate_size + self.embedding_dim
        output_vector_dim = self.slate_size * self.embedding_dim
        condition_dim = self.slate_size + self.embedding_dim

        self.encoder = VariationalEncoder(input_dim=encoded_vector_dim,
                                          hidden_dims=encoder_hidden_dims,
                                          latent_dim=latent_dim)
        self.prior = VariationalEncoder(input_dim=condition_dim,
                                        hidden_dims=prior_hidden_dims,
                                        latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=output_vector_dim,
                               latent_dim=latent_dim,
                               hidden_dims=decoder_hidden_dims)

    def forward(self,
                slate: torch.Tensor,
                responses: torch.Tensor,
                history_items: torch.Tensor,
                history_len: torch.Tensor):

        user_embedding = torch.sum(self.item_embeddings(history_items), dim=1) / history_len.unsqueeze(dim=1)
        slate_embedding = self.item_embeddings(slate).view(-1, self.slate_size * self.embedding_dim)
        condition_vector = torch.cat([responses, user_embedding], dim=1)
        encoded_vector = torch.cat([slate_embedding, condition_vector], dim=1)

        mu_z, log_std_z = self.encoder(encoded_vector)
        z = self.reparameterize(mu_z, log_std_z)
        recon_vector = self.decoder(z)

        mu_prior, log_std_prior = self.prior(condition_vector)
        return mu_z, log_std_z, recon_vector, mu_prior, log_std_prior

    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor):

        std = torch.exp(log_std)
        eps = torch.rand_like(std, device=self.device)

        return mu + eps * std

    @staticmethod
    def kl_loss(mu_z: torch.Tensor, log_std_z: torch.Tensor, mu_prior: torch.Tensor, log_std_prior: torch.Tensor):

        tr_sigmas = (2 * log_std_z).exp() / (2 * log_std_prior).exp()
        mu_sigma_mu = (mu_z - mu_prior) ** 2 / (2 * log_std_prior).exp()
        det_sigmas = 2 * (log_std_prior - log_std_z)

        divergence = tr_sigmas + mu_sigma_mu + det_sigmas
        divergence -= 1
        divergence *= 0.5

        return divergence.sum()

    def recon_loss(self, slate: torch.Tensor, recon_vector: torch.Tensor):

        recon_loss_function = nn.CrossEntropyLoss()
        probs = self._get_item_probs(recon_vector)
        probs = probs.view(-1, self.num_items + 1)
        slate = slate.view(-1, )

        return recon_loss_function(probs, slate)

    def inference(self, history_items: torch.Tensor, history_len: torch.Tensor):

        user_embedding = torch.sum(self.item_embeddings(history_items), dim=1) / history_len.unsqueeze(dim=1)
        responses = torch.ones((user_embedding.shape[0], self.slate_size)).to(self.device)
        condition_vector = torch.cat([responses, user_embedding], dim=1)

        mu_prior, log_std_prior = self.prior(condition_vector)
        z = self.reparameterize(mu_prior, log_std_prior)
        recon_vector = self.decoder(z)
        probs = self._get_item_probs(recon_vector)
        masking = torch.zeros([probs.shape[0], probs.shape[2]], device=self.device, dtype=torch.float32)
        masking = masking.scatter_(1, history_items, float('-inf'))

        slates = []

        for slate_item in range(self.slate_size):
            slate_output = probs[:, slate_item, :]
            slate_output = slate_output + masking
            slate_item = torch.argmax(slate_output, dim=1)
            slates.append(slate_item)
            masking = masking.scatter_(1, slate_item.unsqueeze(dim=1), float('-inf'))

        return torch.stack(slates, dim=1)

    def _get_item_probs(self, recon_vector: torch.Tensor):
        all_item_embeddings = self.item_embeddings(self.all_items).T
        recon_vector = recon_vector.view(-1, self.slate_size, self.embedding_dim)
        probs = torch.matmul(recon_vector, all_item_embeddings)

        return probs

    def recommend(self, test_loader: DataLoader, topk: int = 10) -> Dict:

        self.eval()

        recommendations = {}
        with torch.no_grad():
            for user_id, _, _, user_history_items, _, history_len in test_loader:
                user_history_items = user_history_items.to(self.device)
                history_len = history_len.to(self.device)

                predicted_slates = self.inference(user_history_items, history_len)
                predicted_slates = predicted_slates.cpu().detach().numpy()
                users = user_id.numpy().tolist()
                recommendations.update(zip(users, map(list, predicted_slates)))

        return recommendations

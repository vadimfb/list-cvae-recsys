from typing import List, Dict
from collections import defaultdict
from time import sleep
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from src.models.list_cvae import ListCVAE
from src.utils.visualize_utils import scatter_plot


class ListCVAETrainer:

    def __init__(self,
                 optimizer_parameters: Dict,
                 model_parameters: Dict,
                 train_parameters: Dict):

        self.model = ListCVAE(**model_parameters)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_parameters)
        self.train_parameters = train_parameters

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        stats = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for _, slate_items, slate_responses, user_history_items, _, history_len in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                slate_items = slate_items.to(self.model.device)
                slate_responses = slate_responses.to(self.model.device).float()
                user_history_items = user_history_items.to(self.model.device)
                history_len = history_len.to(self.model.device)

                mu_z, log_std_z, recon_vector, mu_prior, log_std_prior = self.model(slate_items,
                                                                                    slate_responses,
                                                                                    user_history_items,
                                                                                    history_len)

                kl_loss = self.model.kl_loss(mu_z, log_std_z, mu_prior, log_std_prior)
                recon_loss = self.model.recon_loss(slate_items, recon_vector)

                loss = kl_loss + recon_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                stats.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        return stats

    def eval_model(self, test_loader: DataLoader):

        self.model.eval()

        hits = 0
        random_hits = 0
        with torch.no_grad():
            for _, slate_items, _, user_history_items, slate_lens, history_len in test_loader:
                slate_items = slate_items.to(self.model.device)
                user_history_items = user_history_items.to(self.model.device)
                slate_lens = slate_lens.to(self.model.device)
                history_len = history_len.to(self.model.device)

                predicted_slates = self.model.inference(user_history_items, history_len)

                for slate, slate_len, predicted_slate in zip(slate_items, slate_lens, predicted_slates):
                    slate = slate.cpu().numpy()[:int(slate_len.item())]
                    predicted_slate = predicted_slates.cpu().numpy()
                    n_intersections = np.intersect1d(slate, predicted_slate).shape[0]
                    random_preds = np.random.randint(low=0, high=self.model.num_items, size=self.model.slate_size)
                    n_random_intersections = np.intersect1d(slate, random_preds).shape[0]
                    hits += n_intersections
                    random_hits += n_random_intersections

            hits /= len(test_loader.dataset)
            hits /= self.model.slate_size
            random_hits /= len(test_loader.dataset)
            random_hits /= self.model.slate_size
        return hits, random_hits

    def train_eval_model(self, train_dataset: Dataset, test_dataset: Dataset):

        epochs = self.train_parameters['epochs']
        batch_size = self.train_parameters['batch_size']

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size, shuffle=True, num_workers=4)

        self.model.to(self.model.device)

        train_results = defaultdict(list)
        test_results = []
        random_results = []
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_epoch(train_loader, epoch)
            test_loss, random_loss = self.eval_model(test_loader)
            train_results[epoch] = train_loss
            test_results.append(test_loss)
            random_results.append(random_loss)
            print(f'P@k on test: {test_loss}; random P@k: {random_loss}')
            sleep(0.1)

        return train_results, test_results, random_results

    def train_model(self, train_dataset: Dataset):

        epochs = self.train_parameters['epochs']

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.train_parameters['batch_size'], shuffle=True, num_workers=4)

        self.model.to(self.model.device)

        train_results = defaultdict(list)
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_epoch(train_loader, epoch)
            train_results[epoch] = train_loss

        return train_results

    @staticmethod
    def plot_test_results(test_results: List[float], figure_path: str):
        x = [i + 1 for i in range(len(test_results))]
        y = test_results
        scatter_plot(x=x, y=y, x_label='epoch', y_label='p@k', legend='List-CVAE', path=figure_path)

    @staticmethod
    def plot_train_results(train_results: Dict[int, List[float]], figure_path: str):

        epochs = max(train_results.keys()) + 1
        y = []
        for key in range(epochs):
            y.append(np.mean(train_results[key]))

        x = [i + 1 for i in range(epochs)]

        scatter_plot(x=x, y=y, x_label='epoch', y_label='avg_loss', legend='train curve', path=figure_path)

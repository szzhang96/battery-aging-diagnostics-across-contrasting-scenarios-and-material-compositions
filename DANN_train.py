"""
DANN Training
source domain - Cell 7
target domain - the rest 7 protocols of Dataset 1 + Cell 1A (Dataset 2) + CS2-35 (Dataset 3)
"""
import h5py
import numpy as np
import re
import os
from torch import nn
from tqdm import tqdm
import torch
import torch.optim as optim
from scipy.io import savemat
import time

# GRL
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# G
class SeqFeatureEncoder(nn.Module):
    def __init__(self, feat_dim=1, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.encoder(x)
        return h


# F
class SeqRegressor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, h_seq):
        y_hat = self.head(h_seq).squeeze(-1)
        return y_hat


# D
class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, h_seq):
        h_global = h_seq.mean(dim=1)
        domain_logit = self.classifier(h_global)
        return domain_logit


class SeqFewShotDANN(nn.Module):
    def __init__(self, feat_dim=1, hidden_dim=128, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.encoder = SeqFeatureEncoder(feat_dim, hidden_dim)
        self.regressor = SeqRegressor(hidden_dim)
        self.domain_disc = DomainDiscriminator(hidden_dim)

    def forward(self, x):
        h_seq = self.encoder(x)
        y_hat = self.regressor(h_seq)
        return y_hat

    def forward_dann(self, x):
        h_seq = self.encoder(x)
        y_hat = self.regressor(h_seq)

        h_grl = GradientReversalLayer.apply(h_seq, self.alpha)
        domain_logit = self.domain_disc(h_grl)
        return y_hat, domain_logit


def extract_number(s):
    """
    extract the number
    """
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[0])
    return 0


def train_model(model, source_data, target_labeled_data, target_unlabeled_data, optimizer, scheduler, lambda_adv=0.5, epochs=500, device='cpu'):
    model.to(device)
    model.train()
    x_s, y_s = source_data  # _s: Source _tl: Target labelled   _tu: Target unlabelled
    x_s, y_s = x_s.to(device), y_s.to(device)

    x_tl = torch.cat([tl for tl, _ in target_labeled_data], dim=1)
    y_tl = torch.cat([yl for _, yl in target_labeled_data], dim=1)
    x_tu = torch.cat(target_unlabeled_data, dim=1)
    x_tl, y_tl = x_tl.to(device), y_tl.to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    train_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        # 1. (Source + Target labelled)
        y_hat_s, dom_s = model.forward_dann(x_s)
        y_hat_tl, _ = model.forward_dann(x_tl)
        loss_reg = mse(y_hat_s, y_s) + mse(y_hat_tl, y_tl)
        # 2. (Target unlabelled)
        _, dom_tu = model.forward_dann(x_tu)
        label_tu = torch.zeros_like(dom_tu)
        # 3. (Target labelled)
        _, dom_tl = model.forward_dann(x_tl)
        label_tl = torch.zeros_like(dom_tl)

        label_s = torch.ones_like(dom_s)
        loss_adv = bce(dom_s, label_s) + bce(dom_tu, label_tu) + bce(dom_tl, label_tl)

        loss = loss_reg + lambda_adv * loss_adv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
    return train_losses


def set_global_seed(seed: int = 1):
    import os, random, torch

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_data(idx, input_data, capacity, data_len, PERCENTAGE, statistical_value, device='cpu'):
    set_global_seed(1)

    mu_input, sigma_input, mu_output, sigma_output = statistical_value
    input_standard = (input_data - mu_input) / sigma_input
    output_standard = (capacity - mu_output) / sigma_output
    if idx != 0:
        num_extraction = int(np.ceil(data_len * PERCENTAGE))
        random_indices = np.random.choice(data_len, size=num_extraction, replace=False)
        random_indices = np.sort(random_indices)
        x_l = input_standard[random_indices]
        y_l = output_standard[random_indices]
        all_indices = np.arange(data_len)
        unselected_indices = np.setdiff1d(all_indices, random_indices)
        unselected_indices = np.sort(unselected_indices)

        x_u = input_standard[unselected_indices]
        x_l = torch.tensor(x_l, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        y_l = torch.tensor(y_l, dtype=torch.float32).unsqueeze(0).to(device)
        x_u = torch.tensor(x_u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        x_all = torch.tensor(input_standard, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        x_l = x_l.permute(0, 2, 1)
        x_u = x_u.permute(0, 2, 1)
        x_all = x_l
        y_all = capacity[random_indices]
    else:
        x_l = torch.tensor(input_standard, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        y_l = torch.tensor(output_standard, dtype=torch.float32).unsqueeze(0).to(device)
        x_l = x_l.permute(0, 2, 1)
        x_all, y_all = x_l, capacity
        x_u = 0

    return x_l, y_l, x_all, y_all, x_u


def main(mat_path_MIT, mat_path_oxford, mat_path_CALCE):
    set_global_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with h5py.File(mat_path_MIT, 'r') as f:
        HI_extraction = f['HI_extraction']
        conditions = list(HI_extraction.keys())

    all_results = {}

    mu_input, sigma_input = None, None
    mu_output, sigma_output = None, None

    PERCENTAGE = 0.01
    losses = []
    models = []
    stastic_results = []
    target_labeled = []
    target_unlabeled = []
    x_test_list = []
    y_test_list = []
    cond_list = []
    with h5py.File(mat_path_MIT, 'r') as f, h5py.File(mat_path_oxford, 'r') as f_o, h5py.File(mat_path_CALCE, 'r') as f_C:
        data = f['HI_extraction']
        protocolNames = list(data.keys())

        for idx, cond in enumerate(conditions):
            cellNames = sorted(list(data[protocolNames[idx]].keys()), key=extract_number)
            data_base = data[protocolNames[idx]][cellNames[0]]['Results'][()]
            input_data = np.mean(data_base[[0, 1, 3], :], axis=0)
            capacity = data_base[-1, :]
            data_len = data_base.shape[1]
            print(f"\nProcessing Condition {cond}, Cell {cellNames[0]}, Cycles: {data_len}")
            cond_list.append(cond)
            if idx == 0:
                mu_input, sigma_input = input_data.mean(), input_data.std()
                mu_output, sigma_output = capacity.mean(), capacity.std()
                np.savez('standardization_parameters.npz', mu_input=mu_input, sigma_input=sigma_input,
                         mu_output=mu_output, sigma_output=sigma_output)
                statistical_value = mu_input, sigma_input, mu_output, sigma_output
                print(f"First battery: mu_in={mu_input:.3f}, sigma_in={sigma_input:.3f}")
                print(f"First battery: mu_out={mu_output:.3f}, sigma_out={sigma_output:.3f}")
                x_s, y_s, x_s_test, y_s_test, _ = process_data(idx, input_data, capacity, data_len, PERCENTAGE, statistical_value, device='cpu')
                x_test_list.append(x_s_test)
                y_test_list.append(y_s_test)
            else:
                x_tl, y_tl, x_t_test, y_t_test, x_tu = process_data(idx, input_data, capacity, data_len, PERCENTAGE, statistical_value, device='cpu')
                target_labeled.append((x_tl, y_tl))
                target_unlabeled.append(x_tu)
                x_test_list.append(x_t_test)
                y_test_list.append(y_t_test)

                source_data = x_s, y_s

        data_oxford = f_o['HI_all_2']
        cap_oxford = f_o['Q_all_2']
        cellNames = list(data_oxford.keys())
        input_data = data_oxford[cellNames[0]][()].flatten() /1e3
        capacity = cap_oxford[cellNames[0]][()].flatten() / 1e3
        data_len = input_data.shape[0]
        PERCENTAGE = 0.1
        x_tl, y_tl, x_t_test, y_t_test, x_tu = process_data(10, input_data, capacity, data_len, PERCENTAGE,statistical_value, device='cpu')
        target_labeled.append((x_tl, y_tl))
        target_unlabeled.append(x_tu)
        x_test_list.append(x_t_test)
        y_test_list.append(y_t_test)
        cond_list.append('oxford')

        data_CALCE = f_C['HI_all_2']
        cap_CALCE = f_C['Q_all_2']
        cellNames = list(data_CALCE.keys())
        input_data = data_CALCE[cellNames[0]][()].flatten()
        capacity = cap_CALCE[cellNames[0]][()].flatten()
        data_len = input_data.shape[0]
        PERCENTAGE = 0.05
        x_tl, y_tl, x_t_test, y_t_test, x_tu = process_data(10, input_data, capacity, data_len, PERCENTAGE,statistical_value, device='cpu')

        target_labeled.append((x_tl, y_tl))
        target_unlabeled.append(x_tu)
        x_test_list.append(x_t_test)
        y_test_list.append(y_t_test)
        cond_list.append('CALCE')

        model = SeqFewShotDANN()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.2)

        ts = time.time()
        train_model(model, source_data, target_labeled, target_unlabeled, optimizer, scheduler, lambda_adv=0.25, epochs=500, device='cpu')
        print(time.time() - ts)

        model.eval()
        all_predictions = []
        with torch.no_grad():
            for idx, (x_test, y_test, cond) in enumerate(zip(x_test_list, y_test_list, cond_list)):

                predict_norm = model(x_test).cpu().numpy().flatten()
                predict_original = predict_norm * sigma_output + mu_output

                result = np.column_stack([y_test, predict_original])
                all_results[f'cond_{cond}'] = result
                stastic_results.append(np.mean(np.abs(y_test - predict_original)))
    all_results['statistical'] = stastic_results
    torch.save(model, f'DANN_model.pth')
    print("All DANN models trained and saved")
    print(losses)
    # -------------------------------
    results = {key: (value.cpu().numpy() if hasattr(value, 'cpu') else value)
               for key, value in all_results.items()}

    savemat('DANN_trainingResults.mat', results)
    print(f"\n✅ Training Results saved")

if __name__ == "__main__":
    root_path = "./data"
    mat_path_MIT = root_path + "/MIT/HI.mat"
    mat_path_oxford = root_path + "/oxford/HI.mat"
    mat_path_CALCE = root_path + "/CALCE/HI.mat"
    main(mat_path_MIT, mat_path_oxford, mat_path_CALCE)

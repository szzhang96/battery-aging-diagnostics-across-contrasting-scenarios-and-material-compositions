"""
MTL train
"""
import time

import h5py
import numpy as np
import re
import os
from torch import nn
from tqdm import tqdm
import torch
import torch.optim as optim
from scipy.io import savemat


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


class MTLModel(nn.Module):
    def __init__(self, feat_dim=1, hidden_dim=128, num_tasks=10):
        super().__init__()
        self.encoder = SeqFeatureEncoder(feat_dim, hidden_dim)
        self.heads = nn.ModuleList([
            SeqRegressor(hidden_dim) for _ in range(num_tasks)
        ])
    def forward(self, x, task_id):
        feat = self.encoder(x)
        out = self.heads[task_id](feat)
        return out


def train_model(
        model,
        task_data,
        lr=1e-3,
        epochs=500,
        device='cpu'):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.2)

    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        total_loss = 0.0

        for task_id, (x, y) in enumerate(task_data):
            x_support = x.permute(0, 2, 1)
            x_support, y_support = x_support.to(device), y.to(device)

            pred = model(x_support, task_id)
            loss = criterion(pred, y_support)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()


def standard_and_tensor(x, y, statistic_val, device='cpu'):
    mu_input, sigma_input, mu_output, sigma_output = statistic_val
    x_standard = (x - mu_input) / sigma_input
    y_standard = (y - mu_output) / sigma_output
    input_data = torch.tensor(x_standard, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
    output_data = torch.tensor(y_standard, dtype=torch.float32).unsqueeze(0).to(device)  # [1,T]
    task_data = (input_data, output_data)
    return task_data, y


def standard_and_tensor_others(x, y, device='cpu'):
    mu_input, sigma_input = x.mean(), x.std()  # 标准化
    mu_output, sigma_output = y.mean(), y.std()
    statistic_val = mu_input, sigma_input, mu_output, sigma_output
    x_standard = (x - mu_input) / sigma_input
    y_standard = (y - mu_output) / sigma_output
    input_data = torch.tensor(x_standard, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
    output_data = torch.tensor(y_standard, dtype=torch.float32).unsqueeze(0).to(device)  # [1,T]
    task_data = (input_data, output_data)
    return task_data, y, statistic_val


def extract_number(s):
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[0])
    return 0


def set_global_seed(seed: int = 1):
    import os, random, torch

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(mat_path_MIT, mat_path_oxford, mat_path_CALCE):
    set_global_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_results = {}

    task_data_list = []
    ground_truth_list = []
    cond_list = []
    with h5py.File(mat_path_MIT, 'r') as f, h5py.File(mat_path_oxford, 'r') as f_o, h5py.File(mat_path_CALCE, 'r') as f_C:
        data = f['HI_extraction']
        protocolNames = list(data.keys())

        all_input, all_output = [], []
        for idx, cond in enumerate(protocolNames):
            cellNames = sorted(list(data[protocolNames[idx]].keys()), key=extract_number)
            data_base = data[protocolNames[idx]][cellNames[0]]['Results'][()]  # 读取base工况
            input_data = np.mean(data_base[[0, 1, 3], :], axis=0)
            capacity = data_base[-1, :]
            all_input.append(input_data)
            all_output.append(capacity)

        mu_input = np.mean(np.concatenate([arr.ravel() for arr in all_input]))
        sigma_input = np.std(np.concatenate([arr.ravel() for arr in all_input]))
        mu_output = np.mean(np.concatenate([arr.ravel() for arr in all_output]))
        sigma_output = np.std(np.concatenate([arr.ravel() for arr in all_output]))
        print(f"All batteries: mu_in={mu_input:.3f}, sigma_in={sigma_input:.3f}")
        print(f"All batteries: mu_out={mu_output:.3f}, sigma_out={sigma_output:.3f}")
        np.savez('standardization_parameters_dataset_1.npz', mu_input=mu_input, sigma_input=sigma_input,
                 mu_output=mu_output, sigma_output=sigma_output)
        statistic_val = mu_input, sigma_input, mu_output, sigma_output

        for idx, cond in enumerate(protocolNames):
            cellNames = sorted(list(data[protocolNames[idx]].keys()), key=extract_number)
            data_base = data[protocolNames[idx]][cellNames[0]]['Results'][()]
            input_data = np.mean(data_base[[0, 1, 3], :], axis=0)
            capacity = data_base[-1, :]
            T = data_base.shape[1]
            print(f"\nProcessing Condition {cond}, Cell {cellNames[0]}, Cycles: {T}")

            task, y = standard_and_tensor(input_data, capacity, statistic_val, device='cpu')

            cond_list.append(cond)
            task_data_list.append(task)
            ground_truth_list.append(y)

        data_oxford = f_o['HI_all_2']
        cap_oxford = f_o['Q_all_2']
        cellNames = list(data_oxford.keys())
        input_data = data_oxford[cellNames[0]][()].flatten() / 1e3
        capacity = cap_oxford[cellNames[0]][()].flatten() / 1e3
        task, y, statistic_val_oxford = standard_and_tensor_others(input_data, capacity, device='cpu')
        task_data_list.append(task)
        ground_truth_list.append(y)
        cond_list.append('oxford')

        data_CALCE = f_C['HI_all_2']
        cap_CALCE = f_C['Q_all_2']
        cellNames = list(data_CALCE.keys())
        input_data = data_CALCE[cellNames[0]][()].flatten()
        capacity = cap_CALCE[cellNames[0]][()].flatten()
        task, y, statistic_val_CALCE = standard_and_tensor_others(input_data, capacity, device='cpu')
        task_data_list.append(task)
        ground_truth_list.append(y)
        cond_list.append('CALCE')

        model = MTLModel()
        print(model.state_dict().keys())
        ts = time.time()
        train_model(
            model,
            task_data_list,
            lr=1e-3,
            epochs=500,
            device='cpu')
        print(time.time() - ts)

        model.eval()
        all_statis = []
        for idx, cond in enumerate(protocolNames):
            cellNames = sorted(list(data[protocolNames[idx]].keys()), key=extract_number)
            all_results_cells = {}
            statistical_results = []
            cell_name = cellNames[0]
            data_base = data[protocolNames[idx]][cell_name]['Results'][()]  # 读取base工况
            hi_data = np.mean(data_base[[0, 1, 3], :], axis=0)
            capacity = data_base[-1, :]
            T = data_base.shape[1]
            print(f"\nProcessing Condition {cond}, Cell {cell_name}, Cycles: {T}")

            input_norm = (hi_data - mu_input) / sigma_input
            output_norm = (capacity - mu_output) / sigma_output

            X_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            X_tensor = X_tensor.permute(0, 2, 1)

            with torch.no_grad():
                output = model(X_tensor, idx).cpu().numpy().flatten()
                pred_result = output * sigma_output + mu_output

            result = np.column_stack([capacity, pred_result])
            all_results_cells[cell_name] = result
            mape = np.mean(np.abs((capacity - pred_result) / capacity)) * 100
            rmse = np.sqrt(np.mean((capacity - pred_result) ** 2)) * 1e3  # mAh
            statistical_results.append([mape, rmse])
            all_statis.append([mape, rmse])
            all_results[f'battery_{cond}'] = all_results_cells

        data_oxford = f_o['HI_all_2']
        cap_oxford = f_o['Q_all_2']
        cellNames = list(data_oxford.keys())
        input_data = data_oxford[cellNames[0]][()].flatten() / 1e3
        capacity = cap_oxford[cellNames[0]][()].flatten() / 1e3
        input_norm = (input_data - statistic_val_oxford[0]) / statistic_val_oxford[1]
        X_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        X_tensor = X_tensor.permute(0, 2, 1)
        with torch.no_grad():
            output = model(X_tensor, idx).cpu().numpy().flatten()
            pred_result = output * statistic_val_oxford[3] + statistic_val_oxford[2]

        result = np.column_stack([capacity, pred_result])
        all_results['oxford'] = result
        mape = np.mean(np.abs((capacity - pred_result) / capacity)) * 100
        rmse = np.sqrt(np.mean((capacity - pred_result) ** 2)) * 1e3
        statistical_results.append([mape, rmse])
        all_statis.append([mape, rmse])

        data_CALCE = f_C['HI_all_2']
        cap_CALCE = f_C['Q_all_2']
        cellNames = list(data_CALCE.keys())
        input_data = data_CALCE[cellNames[0]][()].flatten()
        capacity = cap_CALCE[cellNames[0]][()].flatten()
        input_norm = (input_data - statistic_val_CALCE[0]) / statistic_val_CALCE[1]
        X_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        X_tensor = X_tensor.permute(0, 2, 1)
        with torch.no_grad():
            output = model(X_tensor, idx).cpu().numpy().flatten()
            pred_result = output * statistic_val_CALCE[3] + statistic_val_CALCE[2]

        result = np.column_stack([capacity, pred_result])
        all_results['CALCE'] = result
        mape = np.mean(np.abs((capacity - pred_result) / capacity)) * 100
        rmse = np.sqrt(np.mean((capacity - pred_result) ** 2)) * 1e3
        statistical_results.append([mape, rmse])
        all_statis.append([mape, rmse])
        all_results_cells['statistic'] = statistical_results
        all_results['all_statis'] = all_statis

    results = {key: (value.cpu().numpy() if hasattr(value, 'cpu') else value)
               for key, value in all_results.items()}

    savemat('MTL_trainingResults.mat', results)

    print(f"\n✅ Training Results saved")
    torch.save(model, 'MTL_model.pth')
    print("Model saved")

if __name__ == "__main__":
    root_path = "./data"
    mat_path_MIT = root_path + "/MIT/HI.mat"
    mat_path_oxford = root_path + "/oxford/HI.mat"
    mat_path_CALCE = root_path + "/CALCE/HI.mat"
    main(mat_path_MIT, mat_path_oxford, mat_path_CALCE)


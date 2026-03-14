"""
DANN TEST
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

from DANN_train import GradientReversalLayer, SeqFeatureEncoder, SeqRegressor, DomainDiscriminator, SeqFewShotDANN


def extract_number(s):
    """
    extract the number
    """
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[0])
    return 0


def load_scaler(scaler_path):
    data = np.load(scaler_path)
    return  data['mu_input'], data['sigma_input'], data['mu_output'], data['sigma_output']


def main(mat_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mu_input, sigma_input, mu_output, sigma_output = load_scaler('standardization_parameters.npz')

    model = torch.load('DANN_model.pth', map_location=device, weights_only=False)

    all_results = {}
    all_statis = []
    with h5py.File(mat_path, 'r') as f:
        data = f['HI_extraction']
        protocolNames = list(data.keys())

        for idx, cond in enumerate(protocolNames):
            if idx == 0:
                continue
            cellNames = sorted(list(data[protocolNames[idx]].keys()), key=extract_number)
            model.eval()
            all_results_cells = {}
            statistical_results = []
            for cell_idx, cell_name in enumerate(cellNames):
                if cell_idx == 0:
                    continue
                data_base = data[protocolNames[idx]][cell_name]['Results'][()]
                hi_data = np.mean(data_base[[0, 1, 3], :], axis=0)
                capacity = data_base[-1, :]
                T = data_base.shape[1]
                print(f"\nProcessing Condition {cond}, Cell {cell_name}, Cycles: {T}")

                input_norm = (hi_data - mu_input) / sigma_input
                output_norm = (capacity - mu_output) / sigma_output

                X_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                X_tensor = X_tensor.permute(0, 2, 1)

                with torch.no_grad():
                    output = model(X_tensor).cpu().numpy().flatten()  # [T,]
                    pred_result = output * sigma_output + mu_output

                result = np.column_stack([capacity, pred_result])  # [T, 2]
                all_results_cells[cell_name] = result
                mape = np.mean(np.abs((capacity - pred_result) / capacity)) * 100
                rmse = np.sqrt(np.mean((capacity - pred_result) ** 2)) * 1e3  # mAh
                statistical_results.append([mape, rmse])
                all_statis.append([mape, rmse])

            all_results_cells['statistic'] = statistical_results
            all_results[f'battery_{cond}'] = all_results_cells
            all_results['all_statis'] = all_statis

    output_file = 'DANN_testResults.mat'

    results = {key: (value.cpu().numpy() if hasattr(value, 'cpu') else value)
               for key, value in all_results.items()}
    savemat(output_file, results)

if __name__ == '__main__':
    mat_path = "xx.mat"
    main(mat_path)


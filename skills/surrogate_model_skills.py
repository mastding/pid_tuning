# Surrogate Model Skills
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from torch.utils.data import Dataset, DataLoader

class GRUModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, mv, pv, sequence_length=10):
        self.sequence_length = sequence_length
        self.mv = mv
        self.pv = pv
        
    def __len__(self):
        return len(self.mv) - self.sequence_length
    
    def __getitem__(self, idx):
        x_mv = self.mv[idx:idx+self.sequence_length]
        x_pv = self.pv[idx:idx+self.sequence_length]
        x = np.stack([x_mv, x_pv], axis=1)
        y = self.pv[idx+self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

def data_quality_check(df: pd.DataFrame) -> Dict:
    mv = df['MV'].values
    pv = df['PV'].values
    
    mv_range = mv.max() - mv.min()
    mv_coverage = mv_range / 100.0 * 100
    
    pv_std = np.std(pv)
    mv_std = np.std(mv)
    
    data_points = len(df)
    
    if mv_coverage < 20:
        quality = "poor"
        recommendation = "MV coverage insufficient, need more excitation"
    elif mv_coverage < 50:
        quality = "fair"
        recommendation = "MV coverage acceptable but limited"
    else:
        quality = "good"
        recommendation = "Data quality is sufficient for training"
    
    return {
        "quality": quality,
        "mv_coverage_percent": mv_coverage,
        "data_points": data_points,
        "pv_std": pv_std,
        "mv_std": mv_std,
        "recommendation": recommendation
    }

def train_surrogate_model(df: pd.DataFrame, model_path: str, config: Dict) -> Dict:
    mv = df['MV'].values
    pv = df['PV'].values
    
    # Normalize
    mv_mean, mv_std = mv.mean(), mv.std()
    pv_mean, pv_std = pv.mean(), pv.std()
    mv_norm = (mv - mv_mean) / (mv_std + 1e-8)
    pv_norm = (pv - pv_mean) / (pv_std + 1e-8)
    
    # Create dataset
    sequence_length = config.get('sequence_length', 10)
    dataset = TimeSeriesDataset(mv_norm, pv_norm, sequence_length)
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    
    # Model
    model = GRUModel(
        input_size=2,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2)
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Training
    epochs = config.get('epochs', 50)
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'mv_mean': mv_mean,
        'mv_std': mv_std,
        'pv_mean': pv_mean,
        'pv_std': pv_std,
        'config': config
    }, model_path)
    
    return {
        'final_loss': losses[-1],
        'training_epochs': epochs,
        'model_path': model_path,
        'success': True
    }

def fast_predict_score(model_path: str, pid_params: Dict, initial_state: Dict, n_steps: int = 200) -> Dict:
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    model = GRUModel(
        input_size=2,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Simulation
    mv_mean = checkpoint['mv_mean']
    mv_std = checkpoint['mv_std']
    pv_mean = checkpoint['pv_mean']
    pv_std = checkpoint['pv_std']
    
    Kp = pid_params['Kp']
    Ki = pid_params['Ki']
    Kd = pid_params['Kd']
    
    setpoint = initial_state.get('setpoint', 50)
    pv_history = [initial_state.get('pv', setpoint)] * 10
    mv_history = [initial_state.get('mv', 50)] * 10
    
    integral = 0
    prev_error = 0
    
    for step in range(n_steps):
        pv_current = pv_history[-1]
        error = setpoint - pv_current
        
        # PID
        proportional = Kp * error
        integral += Ki * error
        derivative = Kd * (error - prev_error)
        
        mv = np.clip(proportional + integral + derivative, 0, 100)
        
        # Predict next PV
        x_mv = np.array(mv_history[-10:])
        x_pv = np.array(pv_history[-10:])
        x_mv_norm = (x_mv - mv_mean) / (mv_std + 1e-8)
        x_pv_norm = (x_pv - pv_mean) / (pv_std + 1e-8)
        x = np.stack([x_mv_norm, x_pv_norm], axis=1)
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        
        with torch.no_grad():
            pv_next_norm = model(x_tensor).item()
        
        pv_next = pv_next_norm * pv_std + pv_mean
        
        pv_history.append(pv_next)
        mv_history.append(mv)
        prev_error = error
    
    # Calculate metrics
    pv_array = np.array(pv_history[10:])
    errors = setpoint - pv_array
    iae = np.sum(np.abs(errors))
    overshoot = max(0, (pv_array.max() - setpoint) / setpoint * 100)
    
    return {
        'IAE': iae,
        'overshoot_percent': overshoot,
        'final_pv': pv_array[-1],
        'settling_achieved': np.abs(errors[-20:]).max() < 0.02 * setpoint
    }

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import random
import numpy as np

# Sprawdzenie dostępności GPU
device_cpu = torch.device('cpu')
device_gpu = torch.device('cuda') if torch.cuda.is_available() else None

print(f"Dostępne urządzenia:")
print(f"CPU: {device_cpu}")
if device_gpu:
    print(f"GPU: {device_gpu}")
else:
    print("GPU nie jest dostępne.")

# Parametry sieci i danych
input_size = 10      # Rozmiar wektora wejściowego
hidden_size = 50     # Rozmiar ukrytego stanu LSTM
output_size = 1      # Rozmiar wyjściowy
sequence_length = 20 # Długość sekwencji
num_layers = 2       # Liczba warstw LSTM
batch_size = 64      # Rozmiar batcha
num_batches = 100    # Liczba batchy do treningu
epochs = 5           # Liczba epok treningowych

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Ustal ziarno losowości na początku skryptu
set_seed(42)

# Generowanie syntetycznego datasetu
def generate_synthetic_data(num_batches, batch_size, sequence_length, input_size):
    data = []
    targets = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, sequence_length, input_size)
        y = torch.randn(batch_size, output_size)
        data.append(x)
        targets.append(y)
    return data, targets

# Definicja modelu LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]              # Pobierz ostatni czas z sekwencji
        out = self.fc(out)
        return out

# Funkcja treningowa
def train_model(device):
    model = SimpleLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data, targets = generate_synthetic_data(num_batches, batch_size, sequence_length, input_size)
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in zip(data, targets):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        print(f"Urządzenie: {device}, Epoka [{epoch+1}/{epochs}], Średni Loss: {avg_loss:.4f}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Czas treningu na {device}: {total_time:.2f} sekund\n")
    return total_time

# Przeprowadzenie benchmarku
times = {}
times['CPU'] = train_model(device_cpu)

if device_gpu:
    times['GPU'] = train_model(device_gpu)
    speedup = times['CPU'] / times['GPU']
    print(f"Przyspieszenie GPU w stosunku do CPU: {speedup:.2f}x")
else:
    print("GPU nie jest dostępne. Benchmark na GPU nie został wykonany.")

# Wizualizacja wyników (opcjonalnie)
devices = list(times.keys())
training_times = list(times.values())

plt.bar(devices, training_times, color=['blue', 'green'])
plt.xlabel('Urządzenie')
plt.ylabel('Czas treningu (sekundy)')
plt.title('Porównanie czasu treningu LSTM na CPU i GPU')
plt.show()
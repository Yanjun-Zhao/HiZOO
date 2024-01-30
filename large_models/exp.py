CUDA_VISIBLE_DEVICES=0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


# Signal parameters
dt = 0.002
Nt = 500
time = np.arange(Nt) * dt

# Frequency axis
df = 0.5
Nf = 201
fre_focus = np.arange(Nf) * df

# Window size
sigma_t = 0.03

# Function to compute WFT
def WFT_wxk(s, dt, fre_focus, sigma_t):
    time = np.arange(len(s)) * dt
    s = hilbert(s)
    s_fre = fft(s)

    alpha = 1 / (2 * sigma_t ** 2)
    Nt = len(s)
    fre = np.concatenate((np.arange(0, Nt // 2), np.arange(-Nt // 2, 0))) / Nt / dt * 2 * np.pi

    WFT_G = np.zeros((len(fre_focus), Nt), dtype=complex)

    for mf, f in enumerate(fre_focus):
        G = np.exp(-(fre - f * 2 * np.pi) ** 2 / (4 * alpha))
        G = G / np.sqrt(alpha / np.pi)
        WFT_G[mf, :] = ifft(s_fre * G)

    return WFT_G

def WFTI_wxk(WFT_G, dt, fre_focus, sigma_t):
    Nf, Nt = WFT_G.shape
    alpha = 1 / (2 * sigma_t**2)
    time = np.arange(Nt) * dt
    WFTI_s = np.zeros(Nt, dtype=complex)
    for mf in range(Nf):
        G = np.sqrt(np.pi / alpha) * np.exp(-(time - fre_focus[mf] / (2 * np.pi))**2 / (4 * alpha))
        WFTI_s += fft(WFT_G[mf, :]) * G
    WFTI_s = ifft(WFTI_s)
    return WFTI_s



# Generate data and labels for 10 samples
num_samples = 1500
all_data_WFT = []
all_signal_WFT = []

for _ in range(num_samples):
    signal = np.cos(2 * np.pi * 25 * time) + np.cos(2 * np.pi * 50 * time)
    data = signal + 0.1 * np.random.randn(Nt)
    signal_WFT = WFT_wxk(signal, dt, fre_focus, sigma_t)
    data_WFT = WFT_wxk(data, dt, fre_focus, sigma_t)
    all_data_WFT.append(data_WFT)
    all_signal_WFT.append(signal_WFT)

# Transform data into the correct format for CNN input
data_real = np.real(all_data_WFT).reshape(num_samples, 1, Nf, Nt)
data_imag = np.imag(all_data_WFT).reshape(num_samples, 1, Nf, Nt)
labels_real = np.real(all_signal_WFT).reshape(num_samples, 1, Nf, Nt)
labels_imag = np.imag(all_signal_WFT).reshape(num_samples, 1, Nf, Nt)

class ComplexDataset(Dataset):
    def __init__(self, data_real, data_imag, labels_real, labels_imag):
        self.data_real = torch.from_numpy(data_real).float()
        self.data_imag = torch.from_numpy(data_imag).float()
        self.labels_real = torch.from_numpy(labels_real).float()
        self.labels_imag = torch.from_numpy(labels_imag).float()

    def __len__(self):
        return self.data_real.size(0)

    def __getitem__(self, idx):
        return (self.data_real[idx], self.data_imag[idx]), (self.labels_real[idx], self.labels_imag[idx])

# Create dataset
dataset = ComplexDataset(data_real, data_imag, labels_real, labels_imag)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define complex convolution layer
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        real, imag = x
        return self.real_conv(real) - self.imag_conv(imag), self.imag_conv(real) + self.real_conv(imag)

# Define complex convolution network  五层
class ComplexConvNet(nn.Module):
    def __init__(self):
        super(ComplexConvNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 16, 3, padding=1)
        self.conv2 = ComplexConv2d(16, 32, 3, padding=1)
        self.conv3 = ComplexConv2d(32, 64, 3, padding=1)
        self.conv4 = ComplexConv2d(64, 64, 3, padding=1)
        self.conv5 = ComplexConv2d(64, 1, 3, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        real, imag = self.conv1(x)
        real, imag = F.relu(real), F.relu(imag)
        real, imag = self.conv2((real, imag))
        real, imag = F.relu(real), F.relu(imag)
        real, imag = self.conv3((real, imag))
        real, imag = F.relu(real), F.relu(imag)
        real, imag = self.dropout(real), self.dropout(imag)
        real, imag = self.conv4((real, imag))
        real, imag = F.relu(real), F.relu(imag)
        real, imag = self.conv5((real, imag))
        return real, imag

# Initialize network and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexConvNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the network
losses = []
num_epochs = 200
for epoch in range(num_epochs):
    total_loss = 0
    for (noisy_real, noisy_imag), (clean_real, clean_imag) in train_loader:
        noisy_real, noisy_imag = noisy_real.to(device), noisy_imag.to(device)
        clean_real, clean_imag = clean_real.to(device), clean_imag.to(device)
        optimizer.zero_grad()
        output_real, output_imag = model((noisy_real, noisy_imag))
        loss = criterion(output_real, clean_real) + criterion(output_imag, clean_imag)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.legend()
plt.show()

# Function to calculate RMSE
def rmse(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean())


# 生成测试信号及其WFT
test_time = np.arange(Nt) * dt
clean_signal = np.cos(2 * np.pi * 25 * test_time) + np.cos(2 * np.pi * 50 * test_time)
clean_signal_WFT = WFT_wxk(clean_signal, dt, fre_focus, sigma_t)
test_data  = clean_signal+ 0.1 * np.random.randn(Nt)
test_data_WFT =  WFT_wxk(test_data, dt, fre_focus, sigma_t)
test_data_real = np.real(test_data_WFT).reshape(1, 1, Nf, Nt)
test_data_imag = np.imag(test_data_WFT).reshape(1, 1, Nf, Nt)
test_data_real = torch.from_numpy(test_data_real).float().to(device)
test_data_imag = torch.from_numpy(test_data_imag).float().to(device)
clean_labels_real = np.real(clean_signal_WFT).reshape(1, 1, Nf, Nt)
clean_labels_imag = np.imag(clean_signal_WFT).reshape(1, 1, Nf, Nt)
clean_labels_real = torch.from_numpy(clean_labels_real).float().to(device)
clean_labels_imag = torch.from_numpy(clean_labels_imag).float().to(device)


# Model evaluation
model.eval()
with torch.no_grad():
    output_real, output_imag = model((test_data_real, test_data_imag))

# Calculate RMSE for the real and imaginary parts
rmse_real = rmse(output_real, clean_labels_real)
rmse_imag = rmse(output_imag, clean_labels_imag)
print(f'Test RMSE (Real part): {rmse_real.item()}')
print(f'Test RMSE (Imaginary part): {rmse_imag.item()}')


output_magnitude = np.abs(output_real.cpu().numpy().squeeze() + 1j * output_imag.cpu().numpy().squeeze())
clean_magnitude = np.abs(clean_labels_real.cpu().numpy().squeeze() + 1j * clean_labels_imag.cpu().numpy().squeeze())

# 时间轴的刻度和标签
time_ticks = np.linspace(0, Nt-1, num=5)  # 创建5个刻度点
time_labels = np.linspace(0, 1, num=5)  # 创建对应0到1之间的5个标签

# 频率轴的刻度和标签
# 此处我们需要从fre_focus数组中找到25Hz和50Hz对应的索引
frequency_indices = [np.argmin(np.abs(fre_focus - frequency)) for frequency in [25, 50]]
frequency_labels = ['25 Hz', '50 Hz']  # 创建对应的标签

# 创建画布和子图
fig, axs = plt.subplots(1, 3, figsize=(15, 15))
# 设定jet色图
#cmap = plt.get_cmap('jet')
# 含噪信号的时频图
axs[0].imshow(np.abs(test_data_WFT), extent=[time[0], time[-1], fre_focus[0], fre_focus[-1]], aspect='auto', origin='lower', cmap='viridis')
axs[0].set_title('Test Noisy Signal (Time-Frequency)')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Frequency [Hz]')


# 测试输出信号的时频图
axs[1].imshow(output_magnitude, extent=[time[0], time[-1], fre_focus[0], fre_focus[-1]], aspect='auto', origin='lower',cmap='viridis')
axs[1].set_title('Test Output Signal (Time-Frequency)')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Frequency [Hz]')

# 干净信号的时频图
axs[2].imshow(clean_magnitude, extent=[time[0], time[-1], fre_focus[0], fre_focus[-1]], aspect='auto', origin='lower', cmap='viridis')
axs[2].set_title('Test Clean Signal (Time-Frequency)')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()


# 在模型评估后计算测试输出与测试输入的复数差值
diff_input = (output_real - test_data_real) + 1j * (output_imag - test_data_imag)
diff_input_magnitude = np.abs(diff_input.cpu().numpy().squeeze())

# 计算测试输出与干净信号的复数差值
diff_clean  = (output_real - clean_labels_real) + 1j * (output_imag - clean_labels_imag)
diff_clean_magnitude = np.abs(diff_clean.cpu().numpy().squeeze())

# 计算干净信号的模值均值和两个差值的模值均值
clean_signal_magnitude_mean = np.mean(np.abs(clean_labels_real.cpu().numpy().squeeze() + 1j * clean_labels_imag.cpu().numpy().squeeze()))
diff_input_magnitude_mean = np.mean(diff_input_magnitude)
diff_clean_magnitude_mean = np.mean(diff_clean_magnitude)

print(f'Clean Signal Magnitude Mean: {clean_signal_magnitude_mean}')
print(f'Test Output vs Test Input Difference Magnitude Mean: {diff_input_magnitude_mean}')
print(f'Test Output vs Clean Signal Difference Magnitude Mean: {diff_clean_magnitude_mean}')



# 创建画布和子图
fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 一行两列

# 差值图像（测试输出与测试输入）
axs[0].imshow(diff_input_magnitude, extent=[time[0], time[-1], fre_focus[0], fre_focus[-1]], aspect='auto', origin='lower', cmap='viridis')
axs[0].set_title('Difference (Test Output vs Test Input)')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Frequency [Hz]')

# 差值图像（测试输出与干净信号）
axs[1].imshow(diff_clean_magnitude, extent=[time[0], time[-1], fre_focus[0], fre_focus[-1]], aspect='auto', origin='lower', cmap='viridis')
axs[1].set_title('Difference (Test Output vs Clean Signal)')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()



# 确保output_real和output_imag是二维的
output_real_2d = output_real.cpu().numpy().squeeze()
output_imag_2d = output_imag.cpu().numpy().squeeze()

# 使用窗口傅里叶反变换将测试输入、测试输出和干净信号从频率域转换回时域
test_input_WFTI = WFTI_wxk(test_data_real.cpu().numpy().squeeze() + 1j * test_data_imag.cpu().numpy().squeeze(), dt, fre_focus, sigma_t)
test_output_WFTI = WFTI_wxk(output_real_2d + 1j * output_imag_2d, dt, fre_focus, sigma_t)
clean_signal_WFTI = WFTI_wxk(clean_labels_real.cpu().numpy().squeeze() + 1j * clean_labels_imag.cpu().numpy().squeeze(), dt, fre_focus, sigma_t)

# 转换为实数部分用于绘图
test_input_WFTI_real = np.real(test_input_WFTI)
test_output_WFTI_real = np.real(test_output_WFTI)
clean_signal_WFTI_real = np.real(clean_signal_WFTI)

# 绘制时域图
plt.figure(figsize=(15, 5))

# 测试输入信号的窗口傅里叶反变换
plt.subplot(1, 3, 1)
plt.plot(time, test_input_WFTI_real, label='Test Input WFTI')
plt.title('Test Input Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# 测试输出信号的窗口傅里叶反变换
plt.subplot(1, 3, 2)
plt.plot(time, test_output_WFTI_real, label='Test Output WFTI')
plt.title('Test Output Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# 干净信号的窗口傅里叶反变换
plt.subplot(1, 3, 3)
plt.plot(time, clean_signal_WFTI_real, label='Clean Signal WFTI')
plt.title('Clean Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()



# 计算测试输出信号与测试输入信号的时域差值
noise_signal_WFTI = test_input_WFTI_real - clean_signal_WFTI_real

# 计算噪声数据和干净数据的信噪比
signal_power = np.mean(clean_signal_WFTI_real ** 2)
noise_power = np.mean(noise_signal_WFTI ** 2)
snr = 10 * np.log10(signal_power / noise_power)

# 输出信噪比
print(f'Signal-to-Noise Ratio (SNR): {snr:.2f} dB')

# 可视化噪声数据
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(time, noise_signal_WFTI, label='Noise Signal WFTI')
plt.title('Noise Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
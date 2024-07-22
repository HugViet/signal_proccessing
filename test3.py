####################  Tín hiệu âm thanh  ################################
from scipy.signal import butter, lfilter, freqz
from scipy.signal import spectrogram
from pathlib import Path
from scipy.signal import find_peaks
from pydub import AudioSegment
from scipy.signal import butter, filtfilt

import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import numpy as np
import io
import time
import os

from gtts import gTTS
from pydub import AudioSegment

start_time = time.time()

os.remove("PCG_apnea_toan_1.wav")

#low pass filter
def lowpass_filter(data, cutoff_freq, fs, order=4,padlen = 1 ):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data,padlen=padlen)
    return filtered_data

def find_peaks_new(data):
    marked_points = []
    if data[0] > data[1]:
        marked_points.append(0)
    for i in range(len(mean_values) - 1):
        # So sánh giá trị của phần tử hiện tại với giá trị của phần tử sau nó
        if data[i] > data[i + 1] and data[i] > data[i - 1] and data[i] > 0:
            marked_points.append(i)
    if data[len(data) - 1] > data[len(data) - 2]:
        marked_points.append(len(data) - 1)
    return marked_points

# # # # Đọc tệp MP3
# # input = ('sample_4.mp3')
# #input = ('snore_cut.mp3')
# #input = ('snore_10s.mp3')
# #input = ('child_breath.mp3')
# input = ('apnea_sample.mp3')
# # # #input = ('snore.mp3')
# # # input = ('snore_sample.mp3')
# # input = ('test_ai.mp3')
# # input = ('child_breath_short.mp3')
# name = Path(input).stem
#
# audio = AudioSegment.from_mp3(input)
# print("Channels:", audio.channels)
# print("Sample width:", audio.sample_width)
# print("Frame rate:", audio.frame_rate)
# print("Frame count:", len(audio))
# sample_rate_old = audio.frame_rate*2
# data_old= np.array(audio.get_array_of_samples())
# time_old = np.arange(0, len(data_old)) / (sample_rate_old)
# ratio = 1000


# # TXT read and convert to numpy
#
# input = 'PCG_2_0_11.txt'
input= os.path.join("C:/Users/ADMIN/Downloads/pillowControl/pillowControl/mqtt/uploads","dataINMP.txt")
try:
    with open(input, "r", encoding="utf-8") as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"Không tìm thấy tệp: {input}")
    exit()
data_old = np.array([int(line.strip()) for line in lines if line.strip().isdigit()])
sample_rate_old = 2000
time_old = np.arange(0, len(data_old)) / (sample_rate_old)
name = Path(input).stem
ratio = 100

wav_file_path = "PCG_apnea_toan_1.wav"
tts = gTTS(text=input, lang='vi')
temp_mp3_path = "temp_audio.mp3"
tts.save(temp_mp3_path)
audio = AudioSegment.from_mp3(temp_mp3_path)
audio.export(wav_file_path, format="wav")
os.remove(temp_mp3_path)

print(f"Path of .wav: {wav_file_path}")

# data_ft=apply_filter(data,lowcut,highcut,fs,order=6)

# Apply low pass filter
cutoff_freq = 120  # Ví dụ: cắt tần số ở 1000 Hz
data_ft_old = lowpass_filter(data_old, cutoff_freq, sample_rate_old)
data_ft = data_ft_old[np.arange(len(data_ft_old)) % ratio == 0]
time_run = np.arange(0, len(data_ft)) / (sample_rate_old/ratio)
sample_rate = sample_rate_old / ratio

# Tìm đỉnh sau khi lọc
ir_up_peaks_of_peaks, _ = find_peaks(data_ft, height=100)

# Chia các đoạn để tính trung bình
data_for_cut = data_ft
absolute_signal = np.abs(data_for_cut)
time_cut = 0.3
segment_length = int(time_cut*sample_rate)  # Độ dài của mỗi đoạn (số lượng mẫu)
num_segments = len(data_for_cut) // segment_length  # Số lượng đoạn
mean_values = []

for i in range(num_segments): # tổng chiều dài / segment legth
    start_idx = i * segment_length
    end_idx = (i + 1) * segment_length
    segment = absolute_signal[start_idx:end_idx]
    mean_value = np.mean(segment)
    mean_values.append(mean_value*2)


# Tìm đỉnh
marked_points = find_peaks_new(data_ft)
shift_amount = time_cut/2
shifted_points = [(idx*time_cut + shift_amount)  for idx in marked_points]
end_time = time.time()
elapsed_time = end_time - start_time
print("Thời gian tính toán : ", format(elapsed_time))

mean_values = np.array(mean_values)
segment_midpoints = np.arange(segment_length // 2, len(data_for_cut), segment_length)
mid_point_time =  segment_midpoints[:len(mean_values)] / (sample_rate)
peaks1,_ = find_peaks(mean_values)
ir_up_peaks_of_peaks = []

# for i in range(len(peaks1)):
#     if  i == 0 and mean_values[peaks1[0]] > 0.5*mean_values[peaks1[1]]:
#         peaks.append(peaks1[0])
#         continue
#     if  i == len(peaks1) -1 and mean_values[peaks1[len(peaks1)-1]] > 0.5*mean_values[peaks1[len(peaks1)-2]]:
#          peaks.append(peaks1[len(peaks1)-1])
#     if i < len(peaks1) - 1 and mean_values[peaks1[i]] > 0.5*mean_values[peaks1[i-1]] and mean_values[peaks1[i]] > 0.5*mean_values[peaks1[i+1]]:
#         peaks.append(peaks1[i])

if len(peaks1) > 0:  # Kiểm tra nếu peaks1 không rỗng
    ir_up_peaks_of_peaks.append(peaks1[0])

#compare = mean_values[peaks1[0]]

if len(peaks1) > 0:
    compare = mean_values[peaks1[0]]
else:
    print("No peaks found in the data.")

for i in range(0,len(peaks1)):
    if mean_values[peaks1[i]] < 0.3 * compare:
        continue
    ir_up_peaks_of_peaks.append(peaks1[i])
    compare = mean_values[peaks1[i]]

# for i in range(1,len(peaks1)):
#     if mean_values[peaks1[i]] > 0.5 * compare and mean_values[peaks1[i]] > 0.5 * mean_values[peaks1[i + 1]]:
#         peaks.append(peaks1[i])
#         compare = mean_values[peaks1[i]]

peaks_time = segment_midpoints[ir_up_peaks_of_peaks] / sample_rate
peak_distance = []

# snoring_intervals = [(1*sample_rate_old, 3*sample_rate_old), (5*sample_rate_old, 7*sample_rate_old), (9*sample_rate_old, 11*sample_rate_old)]
# not_snoring_intervals = [(0, 1*sample_rate_old), (3*sample_rate_old, 5*sample_rate_old), (7*sample_rate_old, 9*sample_rate_old), (11*sample_rate_old, 12*sample_rate_old)]


# snoring_intervals = [(0, 0.75),(3.25, 4), (4.75, 5.25),(5.75, 6.25)]
# not_snoring_intervals = [(0.75, 3.25), (4, 4.75), (5.25, 5.75), (6.25, 14)]
#
# plt.figure(figsize=(14, 6))
# plt.plot(time_old, data_old,  linewidth=1)
#
# def color_intervals(intervals, color1, color2, label):
#     for start, end in intervals:
#         sub_interval_length = 0.25  # chiều dài mỗi phân đoạn nhỏ
#         num_subintervals = int((end - start) / sub_interval_length)
#         for i in range(num_subintervals + 1):
#             sub_start = start + i * sub_interval_length
#             sub_end = min(sub_start + sub_interval_length, end)
#             color = color1 if i % 2 == 0 else color2
#             plt.axvspan(sub_start, sub_end, color=color, alpha=0.5, label=label if i == 0 else "")
#             label = None  # Chỉ hiển thị nhãn một lần
#
# color_intervals(snoring_intervals, 'lightgreen',  'lightgreen','Ngáy')
# color_intervals(not_snoring_intervals, 'lightpink','lightpink' ,'Không ngáy')
#
# plt.xlabel('Thời gian (s)')
# plt.ylabel('Cường độ')
# plt.title('Tín hiệu âm thanh')
# plt.legend()
# plt.grid(True)
# plt.show()

#khoang cach dinh
for i in range(len(peaks_time)-1):
    distance = peaks_time[i+1] - peaks_time[i]
    peak_distance.append(distance)

print("số lượng đỉnh : ", len(ir_up_peaks_of_peaks)-1)
#print("Khoảng cách các đỉnh ",peak_distance)

count_apnea = 0
for i in peak_distance:
    if i > 5:
        count_apnea += 1
        # print(i)
print("Số lần ngưng thở : ",count_apnea)

fig, axs = plt.subplots(4, 1, figsize=(12,6) ,sharex = True)
fig.suptitle(f"Biểu đồ {name}")

#vi tri cac do thi
ord_data_old = 0 #do thi data goc
ord_data_ft = 1 #do thi data qua low_pass_filter
ord_data_decrase = 2 #do thi data sau khi giam tan so lay mau
ord_chev = 4 #do thi khi qua bo loc chevbyshev
ord_mid_point = 3 # do thi sau khi layt khoang trung binh

axs[ord_data_old].plot(time_old, data_old)
axs[ord_data_old].set_ylabel("Cường độ")
axs[ord_data_old].set_title("Tín hiệu gốc")
# axs[ord_data_old].set_xlabel("Thời gian (s)")

chunk_size= 0.5
axs[ord_data_ft].plot(time_old, data_ft_old)
axs[ord_data_ft].set_ylabel("Cường độ")
axs[ord_data_ft].set_title("Tín hiệu sau khi lọc qua Low pass filter")
# axs[ord_data_ft].set_title("Hình dạng sau xử lý của tín hiệu")
# axs[ord_data_ft].set_xlabel("Thời gian (s)")
# num_chunks = int(len(data_ft_old) / (chunk_size*87000))
# for i in range(num_chunks):
#     axs[ord_data_ft].axvline(x=i * chunk_size, color='grey', linestyle='--', linewidth=1)
# # axs[ord_data_ft].plot(mid_point_time, mean_values, "x", markersize=7)
# axs[ord_data_ft].plot(mid_point_time, mean_values, color='pink', label="Trung bình các đoạn 1s")
# # axs[ord_data_ft].plot(peaks_time, mean_values[ir_up_peaks_of_peaks], "o", markersize=7)

axs[ord_data_decrase].plot(time_run, data_ft)
axs[ord_data_decrase].set_ylabel("Cường độ")
axs[ord_data_decrase].set_title("Tín hiệu sau khi giảm tần số")
dcr_peaks,__=find_peaks(data_ft)
# axs[ord_data_decrase].set_xlabel("Thời gian (s)")
#axs[ord_data_decrase].plot(dcr_peaks/sample_rate, data_ft[dcr_peaks], "x"

#test bộ lọc
#chebyshev loại 1
cutoff = 2
order = 3
ripple = 1  # Độ gợn sóng
data_chev, a = signal.butter(order, cutoff / (0.5 * sample_rate), btype='low')
#data_chev, a = signal.cheby1(order, ripple, cutoff / (0.5 * sample_rate), btype='low')
data_chev = signal.filtfilt(data_chev, a, data_ft)

abs_data_chev = np.abs(data_chev)

#axs[ord_chev].plot(time_run, abs_data_chev, label='Tín hiệu đã lọc Chebyshev loại I')

# axs[ord_chev].plot(time_run, abs_data_chev, label='ButterWorth')
# axs[ord_chev].set_ylabel("Amplitude")
# #axs[ord_chev].set_title("Chebyshev type 1")
# axs[ord_chev].set_title("ButterWorth")


axs[ord_mid_point].plot(mid_point_time, mean_values, color='pink', label="Trung bình các đoạn 1s")
axs[ord_mid_point].plot(peaks_time, mean_values[ir_up_peaks_of_peaks], "x")  # Đánh dấu các đỉnh bằng dấu "x"
axs[ord_mid_point].set_ylabel("Cường độ")
axs[ord_mid_point].set_xlabel("Thời gian (s)")
axs[ord_mid_point].set_title("Kết quả sau xử lý")

#frequency-gram
'''
plt.figure()
freqs_filtered, spectrum_filtered = plt.psd(data, Fs=sample_rate, NFFT=1024, color='red', label='Tín hiệu sau khi lọc')

plt.title('Biểu đồ tần số của tín hiệu âm thanh và sau khi áp dụng bộ lọc thông thấp')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Mật độ năng lượng (dB)')
plt.legend()
plt.grid(True)
'''

#spectrogram
'''
frequencies, times, Sxx = spectrogram(data, fs=sample_rate, nperseg=64, noverlap=32)
Sxx[Sxx == 0] = 1e-10
# Hiển thị spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.colorbar(label='dB')  # Thêm thang màu để biểu thị cường độ
plt.ylabel('Tần số (Hz)')
plt.xlabel('Thời gian (giây)')
plt.title(f'Spectrogram của tệp {name}')
'''

# plt.figure(figsize=(14,4))
# plt.plot(time_old, data_old)
# plt.ylabel("Cường độ")
# plt.title("Tín hiệu gốc")
# plt.xlabel("Thời gian (s)")
#
# plt.figure(figsize=(14,4))
# plt.plot(time_old, data_ft_old)
# plt.ylabel("Cường độ")
# plt.title("Tín hiệu sau khi lọc qua Low pass filter")
# # axs[ord_data_ft].set_title("Hình dạng sau xử lý của tín hiệu")
# plt.xlabel("Thời gian (s)")
#
# fig, axs = plt.subplots(2, 1, figsize=(14,8) ,sharex = True)
# axs[0].plot(time_old, data_old)
# axs[0].set_ylabel("Cường độ")
# axs[0].set_title("Tín hiệu gốc")
# axs[0].set_xlabel("Thời gian (s)")
#
# axs[1].plot(mid_point_time, mean_values, color='pink', label="Trung bình các đoạn 1s")
# axs[1].plot(peaks_time, mean_values[ir_up_peaks_of_peaks], "x")  # Đánh dấu các đỉnh bằng dấu "x"
# axs[1].set_ylabel("Cường độ")
# axs[1].set_title("Kết quả sau xử lý")
# axs[1].set_xlabel("Thời gian (s)")
#
# plt.figure(figsize=(14,4))
# plt.plot(time_run, data_ft)
# plt.ylabel("Cường độ")
# plt.title("Tín hiệu sau khi giảm tần số")
# plt.xlabel("Thời gian (s)")
# plt.plot(dcr_peaks/sample_rate, data_ft[dcr_peaks], "x")
#
#
# plt.figure(figsize=(14,4))
# plt.plot(mid_point_time, mean_values, color='pink', label="Trung bình các đoạn 1s")
# plt.plot(peaks_time, mean_values[ir_up_peaks_of_peaks], "x")  # Đánh dấu các đỉnh bằng dấu "x"
# plt.ylabel("Amplitude")
# plt.xlabel("Thời gian (s)")
# plt.title("Tìm các đỉnh tín hiệu")
#
#
plt.tight_layout()
plt.show()

###################### Tín hiệu PPG #############################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.signal as signal
import csv
import pandas as pd

def find_peaks_of_signal(data, height=None, distance=None):
    # Tìm các đỉnh trong tín hiệu
    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks

def find_peaks_of_peaks(data, height=None, distance=None):
    # Tìm các đỉnh ban đầu
    peaks = find_peaks_of_signal(data, height=height, distance=distance)
    peak_values = data[peaks]
    peaks_of_peaks = find_peaks_of_signal(peak_values, height=height, distance=distance)

    distances = np.diff(peaks_of_peaks)
    return peaks, peaks_of_peaks

ppg_red_data = []
ppg_ir_data = []

fs1 = 250
stride = 1

# def movmean1(A, k):
#     x = A.rolling(k,min_periods= 1, center= True).mean().to_numpy()
#     return x
# def movmedian1(A, k):
#     x = A.rolling(k, min_periods= 1, center= True).median().to_numpy()
#     return

#file_name1='IR_RED_LED.csv'
# file_name = "PPG_2_0_26.txt"
# file_name = "PPG_2_0_11.txt"
file_name= os.path.join("C:/Users/ADMIN/Downloads/pillowControl/pillowControl/mqtt/uploads","dataMAX.txt")

if lines[0].strip() != "ir,red":
    # Thêm "ir,red" vào đầu danh sách các dòng
    lines.insert(0, "ir,red\n")


colum_red = "red"
colum_ir = "ir"
ppg_ir_data = pd.read_csv(file_name, usecols= [colum_ir]).to_numpy()
ppg_red_data = pd.read_csv(file_name,usecols= [colum_red]).to_numpy()
ppg_red_data_smoothed = ppg_red_data
ppg_ir_data_smoothed = ppg_ir_data
ppg_time = np.arange(0, len(ppg_ir_data)) / (fs1)
ppg_red_time = np.arange(0, len(ppg_red_data)) / (fs1)

fig, axs = plt.subplots(2, 1, figsize=(12,8) ,sharex = True)
#vi tri cac do thi
red_data = 0
ir_data = 1
axs[red_data].plot(ppg_time[:1000], ppg_red_data_smoothed[:1000], label='PPG Red')
axs[red_data].set_title("PPG RED Data")
axs[red_data].set_xlabel('Thời gian (giây)')
axs[red_data].set_ylabel('Biên độ')

axs[ir_data].plot(ppg_time[:1000], ppg_ir_data_smoothed[:1000], label='PPG IR')
axs[ir_data].set_title("PPG IR Data")
axs[ir_data].set_xlabel('Thời gian (giây)')
axs[ir_data].set_ylabel('Biên độ')
plt.show()

####################################################### NHip tim ###########################################
ppg_ir_data=ppg_ir_data.flatten()
ppg_red_data=ppg_red_data.flatten()
ir_up_peaks_of_peaks, ir_up_peaks= find_peaks_of_peaks(ppg_ir_data, distance=10)
red_up_peaks_of_peaks, red_up_peaks= find_peaks_of_peaks(ppg_red_data, distance=10)
ir_low_peaks_of_peaks, ir_low_peaks= find_peaks_of_peaks(-ppg_ir_data, distance=10)
red_low_peaks_of_peaks, red_low_peaks= find_peaks_of_peaks(-ppg_red_data, distance=10)
#tinh khoang cach dinh ra nhijp tim,
RR = ir_up_peaks_of_peaks[ir_up_peaks[1:]] - ir_up_peaks_of_peaks[ir_up_peaks[:-1]]
# RR = ampl[1:] - ampl[:-1]
FHR = 60 * fs1 / RR
#print("nhịp tim : ",FHR) #nhịp tim

FHR_time = ir_up_peaks_of_peaks[ir_up_peaks] / fs1
FHR_time = np.delete(FHR_time, -1)
ppg_ir_peaks_time = ir_up_peaks_of_peaks / fs1
ppg_ir_up_peaks_of_peaks_time = ir_up_peaks_of_peaks[ir_up_peaks] / fs1
ppg_ir_low_peaks_of_peaks_time = ir_low_peaks_of_peaks[ir_low_peaks] / fs1
ppg_red_peaks_time = red_up_peaks_of_peaks / fs1
ppg_red_up_peaks_of_peaks_time = red_up_peaks_of_peaks[red_up_peaks] / fs1
ppg_red_low_peaks_of_peaks_time = red_low_peaks_of_peaks[red_low_peaks] / fs1

# #Các biểu đồ
# plt.figure(figsize=(9, 3))
# plt.plot(ppg_time, ppg_ir_data, label='Original Signal')
# plt.plot(ppg_ir_peaks_time, ppg_ir_data[ir_up_peaks_of_peaks], 'x', label='Peaks')
# plt.plot(ppg_ir_up_peaks_of_peaks_time, ppg_ir_data[ir_up_peaks_of_peaks][ir_up_peaks], 'o', label='Peaks of Peaks')
# plt.legend()
# plt.show()

# vẽ biểu đồ và tìm đỉnh
fig, axs = plt.subplots(2, 1, figsize=(14,10) ,sharex = True)
axs[0].plot(ppg_time, ppg_red_data_smoothed)
axs[0].plot(ppg_red_peaks_time, ppg_red_data[red_up_peaks_of_peaks], 'x', label='Peaks')
axs[0].plot(ppg_red_up_peaks_of_peaks_time,  ppg_red_data[red_up_peaks_of_peaks][red_up_peaks],'o', label='Upper Envelope')
# axs[0].plot(ppg_red_low_peaks_of_peaks_time, ppg_red_data[red_low_peaks_of_peaks][red_low_peaks],'o', label='Lower Envelope')
axs[0].set_xlabel('Thời gian (giây)')
axs[0].set_ylabel('Biên độ')
axs[0].set_title("PPG RED Data")

axs[1].plot(ppg_time, ppg_ir_data_smoothed)
plt.plot(ppg_ir_peaks_time, ppg_ir_data[ir_up_peaks_of_peaks], 'x', label='Peaks')
axs[1].plot(ppg_ir_up_peaks_of_peaks_time, ppg_ir_data[ir_up_peaks_of_peaks][ir_up_peaks],'o', label='Upper Envelope')
# axs[1].plot(ppg_ir_low_peaks_of_peaks_time, ppg_ir_data[ir_low_peaks_of_peaks][ir_low_peaks],'o', label='Lower Envelope')
axs[1].set_xlabel('Thời gian (giây)')
axs[1].set_ylabel('Biên độ')
axs[1].set_title("PPG IR Data")

plt.legend()
plt.show()

x = np.arange(1, len(ppg_ir_data) + 1)


#################################################### SPO2 ########################################
# Sliding window to obtain the maximum and minimum of the AC component
window_size = int(fs1*0.75)
upper_envelope_red = []
lower_envelope_red = []
upper_envelope_ir = []
lower_envelope_ir = []

for i in range(0, len(ppg_red_data_smoothed) - window_size, stride):
    window_red = ppg_red_data_smoothed[i:i+window_size]
    window_ir = ppg_ir_data_smoothed[i:i+window_size]
    upper_envelope_red.append(np.max(window_red))
    lower_envelope_red.append(np.min(window_red))
    upper_envelope_ir.append(np.max(window_ir))
    lower_envelope_ir.append(np.min(window_ir))

ppg_red_data=np.array(ppg_red_data).flatten()
ppg_ir_data=np.array(ppg_ir_data).flatten()

up_peaks_red,__0=find_peaks(ppg_red_data)
low_peaks_red,__1=find_peaks(-ppg_red_data)
up_peaks_ir,__2=find_peaks(ppg_ir_data)
low_peaks_ir,__3=find_peaks(-ppg_ir_data)

fig, axs = plt.subplots(2, 1, figsize=(14,7) ,sharex = True)

axs[0].plot(ppg_time, ppg_red_data_smoothed)
axs[0].plot(ppg_red_up_peaks_of_peaks_time,  ppg_red_data[red_up_peaks_of_peaks][red_up_peaks], label='Upper Envelope')
axs[0].plot(ppg_red_low_peaks_of_peaks_time, ppg_red_data[red_low_peaks_of_peaks][red_low_peaks], label='Lower Envelope')
axs[0].set_xlabel('Thời gian (giây)')
axs[0].set_ylabel('Biên độ')
axs[0].set_title("PPG RED Data")

axs[1].plot(ppg_time, ppg_ir_data_smoothed)
axs[1].plot(ppg_ir_up_peaks_of_peaks_time, ppg_ir_data[ir_up_peaks_of_peaks][ir_up_peaks], label='Upper Envelope')
axs[1].plot(ppg_ir_low_peaks_of_peaks_time, ppg_ir_data[ir_low_peaks_of_peaks][ir_low_peaks], label='Lower Envelope')
axs[1].set_xlabel('Thời gian (giây)')
axs[1].set_ylabel('Biên độ')
axs[1].set_title("PPG IR Data")

# fig, axs = plt.subplots(2, 1, figsize=(16,8) ,sharex = True)
# axs[red_data].plot(ppg_red_data_smoothed, label='PPG Red')
# axs[red_data].plot(upper_envelope_red, label='upper Red')
# axs[red_data].plot(lower_envelope_red, label='low Red')
# axs[red_data].set_title("PPG RED Data")
#
# axs[ir_data].plot(ppg_ir_data_smoothed, label='PPG IR')
# axs[ir_data].plot(upper_envelope_ir, label='up ir')
# axs[ir_data].plot(lower_envelope_ir, label='low ir')
# axs[ir_data].set_title("PPG IR Data")

plt.tight_layout()
plt.legend()
plt.show()

upper_envelope_red_np = np.array(upper_envelope_red)
lower_envelope_red_np = np.array(lower_envelope_red)
upper_envelope_ir_np = np.array(upper_envelope_ir)
lower_envelope_ir_np = np.array(lower_envelope_ir)

AC_red_range = upper_envelope_red_np - lower_envelope_red_np
AC_ir_range = upper_envelope_ir_np - lower_envelope_ir_np
# AC_red_range = ppg_red_data[red_up_peaks_of_peaks][red_up_peaks] - ppg_red_data[red_low_peaks_of_peaks][red_low_peaks]
# AC_ir_range = ppg_ir_data[ir_up_peaks_of_peaks][ir_up_peaks] - ppg_ir_data[ir_low_peaks_of_peaks][ir_low_peaks]

epsilon = 1e-10

DC_red_range = lower_envelope_red_np + epsilon
DC_ir_range = lower_envelope_ir_np + epsilon

r = (AC_red_range / DC_red_range) / (AC_ir_range / DC_ir_range)

# Biểu đồ Nhịp tim
plt.figure(figsize=(9, 3))
plt.plot(FHR_time, FHR)
plt.title("Nhịp tim")
plt.xlabel('Thời gian (giây)')
plt.ylabel('Nhịp tim (nhịp/phút)')
plt.tight_layout()
plt.show()

spo2 = 110 - 25*r
spo2_time= np.arange(0, len(spo2)) / (fs1)
# Plotting the SpO2 values
plt.figure(figsize=(16, 7))
plt.title("Nồng độ Oxi trong máu")
plt.plot(spo2_time,spo2, label='SpO2')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Nồng độ (%)')
plt.legend()
plt.show()


#------------------------------------------------------------------------------------------------------
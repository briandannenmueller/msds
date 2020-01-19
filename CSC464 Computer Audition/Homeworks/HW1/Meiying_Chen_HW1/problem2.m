%% Homework 1 Problem 2
% Author: Meiying(Melissa) Chen
% Date: Sep 8, 2019

clear, clc, close all;

%% 2-b
[y_my,Fs_my] = audioread('myvoice.wav');                 % read 'myvoice.wav' into MATLAB
y_my = y_my - mean(y_my);                                % zero mean
rms_my = rms(y_my);                                      % calculate RMS value

samples = [1,length(y_my)];                              % read 'noise.wav' with the same length
[y_noise,Fs_noise] = audioread('noise.wav',samples);
y_noise = y_noise - mean(y_noise);                       % zero mean
rms_noise = rms(y_noise);                                % calculate RMS value

% scale myvoice SNR to 5db
sn = snr(y_my*3.11955, y_noise);
y_scale = y_my*3.11955;
rms_scale = rms(y_scale);
audiowrite('myvoice_noisy.wav',y_scale + y_noise,Fs_my)  % write the mixed audio to wav file

%% 2-c
fs = 16000;
y_down = resample(y_my,fs,44100);                        % downsample to 16kHz
N = 512;                                                 % sample size 512
y_slice = y_down(11353:11352+N,:);
% apply hamming window
w = hamming(N);  
y_w = w'.*y_slice;
% apply fft with 4 times zero bedding
fft_len = N*4;
y_fft = fft(y_w, fft_len); 
% calculate the real frequency resolution
f_temp = fs/2*linspace(0,1,N/2+1);
resl = f_temp(2) - f_temp(1);
% plot
figure(1);
plot((0:fft_len/2)*fs/fft_len, 20*log10(abs(y_fft(1:fft_len/2+1))));
grid on;
xlabel('frequency (Hz)');
ylabel('amplitude (dB)');

%% 2-d
% apply stft
w = hamming(512);
noverlap = 256;
nfft = 512;
[s, f, t] = spectrogram(y_down, w, noverlap, nfft, fs);

% plot
figure(2);
imagesc(t,f,20*log10(abs(s)));
xlabel('time (second)');
ylabel('frequency (Hz)');
set(gca,'YDir','normal');
colorbar

%% 2-e
[s, f, t] = stft(y_down,fs,'Window',w,'OverlapLength',noverlap,'FFTLength',nfft);
% zero linear-amplitude spectra outside the frequency range [300Hz,3400Hz]
f_phone = (f >= 300 & f <= 3400);
s_phone = f_phone.*s;
% recuild with Overlap-Add technique
[y_rebuilt,t_rebuilt] = istft(s_phone,fs,'Window',w,'OverlapLength',noverlap,'FFTLength',nfft,'Method','ola','ConjugateSymmetric',true);
% write to file
audiowrite('myvoice_telephone.wav',y_rebuilt,fs)  

% zero phase and resuld
s_0phase = real(s_phone);
[y_0phase,t_0phase] = istft(s_0phase,fs,'Window',w,'OverlapLength',noverlap,'FFTLength',nfft,'Method','ola','ConjugateSymmetric',true);
% write to file
audiowrite('myvoice_telephone_zerophase.wav',y_0phase,fs)











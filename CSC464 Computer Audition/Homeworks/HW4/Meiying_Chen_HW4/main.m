clear, clc, close all;


%% 1-b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read audio
[x, fs] = audioread('piano.wav');

% get original spectrogram
win_len = 2048;
hop = 512;
nfft = 2048;
[V,f,t] = stft(x, win_len, hop, nfft, fs);

% apply NMF
[m, n] = size(V);
r = 4;
bUpdateH = 1;
bUpdateW = 1;
initW = 1 + rand(m, r);
initH = 1 + rand(r, n);
nIter = 50;
[W, H, KL] = myNMF(V, r, nIter, initW, initH, bUpdateW, bUpdateH);

% reconstruct
V_rec = W*H.*(exp(1j*angle(V)));
x_rec = istft(V_rec, nfft, win_len, hop);

% write into file
audiowrite('piano_recon_r4.wav', x_rec/max(abs(x_rec)), fs)

% plot
plotNMF(V, V_rec, x, x_rec, W, H, r, fs, t)





%% 1-c %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read audio
[x, fs] = audioread('piano.wav');
win_len = 2048;
hop = 512;
nfft = 2048;
[V,f,t] = stft(x, win_len, hop, nfft, fs);
% apply NMF
[m, n] = size(V);
r = 5;
bUpdateH = 1;
bUpdateW = 1;
initW = 1 + rand(m, r);
initH = 1 + rand(r, n);
nIter = 50;
[W, H, KL] = myNMF(V, r, nIter, initW, initH, bUpdateW, bUpdateH);
% reconstruct
V_rec = W*H.*(exp(1j*angle(V)));
x_rec = istft(V_rec, nfft, win_len, hop);
% write into file
audiowrite('piano_recon_r5.wav', x_rec/max(abs(x_rec)), fs)
% plot
plotNMF(V, V_rec, x, x_rec, W, H, r, fs, t)



%% 1-d %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x, fs] = audioread('speech_train.wav');
% get original spectrogram
win_len = 1024;
hop = 512;
nfft = 1024;
[V,f,t] = stft(x, win_len, hop, nfft, fs);
% apply NMF
[m, n] = size(V);
r = 200;
bUpdateH = 1;
bUpdateW = 1;
initW = 1 + rand(m, r);
initH = 1 + rand(r, n);
nIter = 20;
[W, H, KL] = myNMF(V, r, nIter, initW, initH, bUpdateW, bUpdateH);
% reconstruct
V_rec = W*H.*(exp(1j*angle(V)));
x_rec = istft(V_rec, nfft, win_len, hop);
audiowrite('speech_train_rec.wav', x_rec/max(abs(x_rec)), fs)
% plot the first 5 dictionary
plotNMF(V, V_rec, x, x_rec, W, H, 5, fs, t)

W_speech = W;



%% 1-e %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x, fs] = audioread('noise_train.wav');
% get original spectrogram
win_len = 1024;
hop = 512;
nfft = 1024;
[V,f,t] = stft(x, win_len, hop, nfft, fs);
% apply NMF
[m, n] = size(V);
r = 200;
bUpdateH = 1;
bUpdateW = 1;
initW = 1 + rand(m, r);
initH = 1 + rand(r, n);
nIter = 20;
[W, H, KL] = myNMF(V, r, nIter, initW, initH, bUpdateW, bUpdateH);
% reconstruct
V_rec = W*H.*(exp(1j*angle(V)));
x_rec = istft(V_rec, nfft, win_len, hop);
audiowrite('noise_train_rec.wav', x_rec/max(abs(x_rec)), fs)
% plot the first 5 dictionary
plotNMF(V, V_rec, x, x_rec, W, H, 5, fs, t)

W_noise = W;


%% 1-f %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x, fs] = audioread('noisyspeech.wav');
% get original spectrogram
win_len = 1024;
hop = 512;
nfft = 1024;
[V,f,t] = stft(x, win_len, hop, nfft, fs);
% apply NMF
[m, n] = size(V);
r = 400;
bUpdateH = 1;
bUpdateW = 0;
initW = [W_speech, W_noise];
initH = 1 + rand(r, n);
nIter = 20;
[W, H, KL] = myNMF(V, r, nIter, initW, initH, bUpdateW, bUpdateH);

% reconstruct the speech
V_speech_rec = W_speech*H(1:200,:).*(exp(1j*angle(V)));
speech_rec = istft(V_speech_rec, nfft, win_len, hop);
audiowrite('speech_sep.wav', speech_rec/max(abs(speech_rec)), fs)

% reconstruct the noise
V_noise_rec = W_speech*H(201:400,:).*(exp(1j*angle(V)));
noise_rec = istft(V_noise_rec, nfft, win_len, hop);
audiowrite('noise_sep.wav', noise_rec/max(abs(noise_rec)), fs)





%% 1-f %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluate separation
se = [speech_rec; noise_rec];
[speech, ~] = audioread('speech_test.wav');
[noise, ~] = audioread('noise_test.wav');
s = [speech(1:length(speech_rec))'; noise(1:length(noise_rec))'];
[SDR,SIR,SAR,perm] = bss_eval_sources(se,s);

disp([SDR,SIR,SAR,perm])


%% 2-b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('pitchdata.mat');
[x, fs] = audioread('female_factory.wav');
% get the magnitude spectrogram
win_len = 1024;
hop = 160;
nfft = win_len * 4;
[s,f,t]= stft(x, win_len, hop, nfft,fs);

% estimate pitch according to max log-likelihood
[~, nframe] = size(loglikeMat);
[~, pitch_index] = max(loglikeMat);
pitch = index2hz(pitch_index);

% plot estimation on the spectrogram
imagesc(t,f,log(abs(s)));
hold on
plot(t,pitch,'.');
set(gca,'YDir','normal');



%% 2-c %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pitch_viterbi = index2hz(myViterbi(transMat,loglikeMat,initProb));

% plot estimation on the spectrogram
imagesc(t,f,log(abs(s)));
hold on
plot(t,pitch_viterbi,'.');
set(gca,'YDir','normal');




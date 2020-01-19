clear, clc, close all;


%% 1-c
win = hamming(512);
hop = 512;
th = 0.11;
frameLen = 512;
frameHop = 512;

[x,fs] = audioread('Pop.wav');
[onsets, onsetStrength] = onset_energy(x, win, hop, th); % calculate onset location
y = synth_onset(x, frameLen, frameHop, onsets); % syncronize with white noise
audiowrite('Pop_onset_energy.wav',y/max(abs(y)),fs);  % write to file




%% 1-e
win = hamming(512);
hop = 512;
th = 0.095;
gamma = 1;
frameLen = 512;
frameHop = 512;

[x,fs] = audioread('Pop.wav');

[onsets, onsetStrength] = onset_spectral(x, win, hop, th, gamma);

y = synth_onset(x, frameLen, frameHop, onsets); % syncronize with white noise
audiowrite('Pop_onset_spectral.wav',y/max(abs(y)),fs);  % write to file


%% 1-f
win = hamming(512);
hop = 512;
th = 0.069;
frameLen = 512;
frameHop = 512;

[x,fs] = audioread('Schumann.wav');
[onsets, onsetStrength] = onset_energy(x, win, hop, th); % calculate onset location
y = synth_onset(x, frameLen, frameHop, onsets); % syncronize with white noise
audiowrite('Shumann_onset_energy.wav',y/max(abs(y)),fs);  % write to file

win = hamming(512);
hop = 512;
th = 0.224;
gamma = 1;
frameLen = 512;
frameHop = 512;

[x,fs] = audioread('Schumann.wav');

[onsets, onsetStrength] = onset_spectral(x, win, hop, th, gamma);
y = synth_onset(x, frameLen, frameHop, onsets); % syncronize with white noise
audiowrite('Schumann_onset_spectral.wav',y/max(abs(y)),fs);  % write to file



win = hamming(512);
hop = 512;
th = 0.63;
frameLen = 512;
frameHop = 512;

[x,fs] = audioread('Haydn.wav');
[onsets, onsetStrength] = onset_energy(x, win, hop, th); % calculate onset location
y = synth_onset(x, frameLen, frameHop, onsets); % syncronize with white noise
audiowrite('Haydn_onset_energy.wav',y/max(abs(y)),fs);  % write to file

win = hamming(512);
hop = 512;
th = 0.33;
gamma = 1000;
frameLen = 512;
frameHop = 512;

[x,fs] = audioread('Haydn.wav');

[onsets, onsetStrength] = onset_spectral(x, win, hop, th, gamma);
y = synth_onset(x, frameLen, frameHop, onsets); % syncronize with white noise
audiowrite('Haydn_onset_spectral.wav',y/max(abs(y)),fs);  % write to file




%% 2-a
winlen = 216;
noverlap = winlen/2;
nfft = 4 * winlen;
[s, f, t] = spectrogram(onsetStrength, winlen, noverlap, nfft, fs/frameLen);

% plot
figure(2);
imagesc(t,f*60,20*log10(abs(s)));
xlabel('Time (second)');
ylabel('Tempo (BPM)');
set(gca,'YDir','normal');
colorbar


%% 3-b
lambda = 1;
tempoExpected = 130;
wavData = 'Pop.wav';

beats = beat_dp(wavData, frameLen, onsetStrength, tempoExpected, lambda);
y = synth_onset(x, frameLen, frameHop, beats);
audiowrite('Pop_beats.wav', y/max(abs(y)),fs)








































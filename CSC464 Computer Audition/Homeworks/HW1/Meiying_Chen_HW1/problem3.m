%% Homework 1 Problem 2
% Author: Meiying(Melissa) Chen
% Date: Sep 9, 2019

clear, clc, close all;

%% 3-a
fs = 44100;                             % sample rate
ts = 1/fs;
t = 0:ts:3-ts;
% generate frequency of sin waves
f = (110);
for k = 2:6
    f(k) = f(k-1) * 2;
end
% generate sin wave
x = sin(2*pi*f(1)*t);
MIDI = (69 + 12*log2(f(1)/440));
for k = 2:6
    xk = sin(2*pi*f(k)*t); 
    MIDI(k) = 69 + 12*log2(f(k)/440);    % calculate MIDI number of each frequency
    x = x + xk;                          % synthesize
end
x_n = x / max(abs(x));                   % normalize to avoid clipping
audiowrite('complextone.wav', x_n, fs)   % write to file


%% 3-b
% apply gaussmf function to each sinusoids
x_g = sin(2*pi*f(1)*t) * gaussmf(MIDI(1),[12 81]);
for k = 2:6
    xk_g = sin(2*pi*f(k)*t) * gaussmf(MIDI(k),[12 81]);
    x_g = x_g + xk_g;                              % synthesize
end
x_g_n = x_g/max(abs(x_g));                          % normalize to avoid clipping
audiowrite('complextone_Gaussian.wav', x_g_n, fs)  % write to file


%% 3-c
t = 0:ts:0.5-ts;                                   % change signal duration to 0.5s
silence = sin(2*pi*0*t);                           % make the silence piece
b = x_g(1:fs*0.5);                                 % slice singal in part b
x_attach = [b silence];                            % make tone 1
% make tone 2-12
for k = 2:12
    MIDI = MIDI + 1;
    f_new = 440 * 2.^ ((MIDI - 69)/12);            % calculate new frequency as MIDI changed
    % generate sin wave with new frequencies
    x_new = sin(2*pi*f_new(1)*t) * gaussmf(MIDI(1),[12 81]);
    for q = 2:6
        xk = sin(2*pi*f_new(q)*t) * gaussmf(MIDI(q),[12 81]);
        x_new = x_new + xk;                        % synthesize                         
    end
    x_attach = [x_attach x_new silence];           % attach tones

end

shepardtone = [x_attach];
for k = 1:4
    shepardtone = [shepardtone x_attach];
end

shepardtone_n = shepardtone / max(abs(shepardtone));  % normalize to avoid clipping

audiowrite('Shepardtone.wav', shepardtone_n, fs)      % write to file























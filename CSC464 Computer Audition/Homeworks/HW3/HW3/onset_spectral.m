function [onsets, onsetStrength] = onset_spectral(x, win, hop, th, gamma)
% Spectral-based onset detection (spectral flux). 
% The spectral flux calculation is based on the compressed magnitude spectrogram: spectrogram_compressed = log(1 + gamma * spectrogram). 
% This is to reduce the dynamic range of the spectrogram. %
% Input
% - x        : audio waveform
% - win      : window function
% - hop      : window hopsize(in samples)
% - th       : threshold to determine onsets
% - gamma    : parameter for spectrogram compression
% Output
% - onsets         : frame indices of the onsets
% - onsetsStrength : normalized onset strength curve, one value per frame,
% range in [0, 1]

    x = x'; % turn column vector into row vector
    nframes = ceil(length(x) / hop);
    x = [x, zeros(1,hop*nframes - length(x))];% zero pad signal to have complete frames

    %function [onsets, onsetStrength] = onset_spectral(x, win, hop, th, gamma)
    [s, ~] = spectrogram(x,win,0,hop,'yaxis');
    s_comp = log(1 + gamma * abs(s)); % compression
    s_diff = sum(diff(s_comp, 1, 2)); % discrete temporal derivative
    s_diff_rec = s_diff .* (s_diff > 0); % half-rectification
    % peak picking
    miu = s_diff_rec;
    halfavewin = 4;
    for n = halfavewin + 1: length(s_diff_rec) - halfavewin
        miu(1, n) = sum(s_diff_rec(n - halfavewin: n+halfavewin)) / (halfavewin + 1);
    end
    ave_diff_rec = s_diff_rec .* ((s_diff_rec - miu) >= 0);
    onsetStrength = ave_diff_rec / max(ave_diff_rec);
    loc = onsetStrength > th;
    onsets = find(loc);
    
    figure(1)
    subplot(2,1,1);
    subplot(gca, 'position', [0.2,0.2,0.32,0.32]);
    plot(onsetStrength);grid on;
    
    subplot(2,1,2);
    subplot(gca, 'position', [0.2,0.2,0.32,0.1]);
    plot(loc);grid on;
    
end
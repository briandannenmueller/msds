function [onsets, onsetStrength] = onset_energy(x, win, hop, th)
% Energy-based onset detection
% 
% Input: 
%   - x       : audio waveform
%   - win     : window function
%   - hop     : window hop size(in samples)
%   - th      : global threshold to determine onsets
% Output:
%   - onsets: frame indices of onsets
%   - onsetsStrength: normalized onsets strength curve, one value per
%   frame, range in [0,1]


    x = x'; % turn column vector into row vector
    nframes = ceil(length(x) / hop);
    x = [x, zeros(1,hop*nframes - length(x))];% zero pad signal to have complete frames


    % calculate local energy
    energy_local = zeros(1, nframes);
    for n = 1:nframes-1
        localsum = 0;
        for m  = 1:hop
            localsum = localsum + ((x(m+n*hop).^2) * win(m));
        end 
        energy_local(1, n) = localsum;
    end                                                                                                                  

    % differentiate then half-wave rectify
    energy_diff = diff(energy_local);
    temp = ((energy_diff + abs(energy_diff)) / 2);
    energy_diff_rec = temp .* (temp > 0);

    % improve with log
    energy_log = diff(log(energy_local));
    temp = ((energy_log + abs(energy_log)) / 2);
    energy_log_rec = temp .* (temp > 0);

    % normalize
    onsetStrength = energy_log_rec / max(energy_log_rec);

    % peak-picking
    loc =  onsetStrength > th; % thresholding
    onsets = find(loc);
    

    figure(1)
    subplot(2,1,1);
    subplot(gca, 'position', [0.2,0.2,0.32,0.32]);
    plot(onsetStrength);grid on;
    
    subplot(2,1,2);
    subplot(gca, 'position', [0.2,0.2,0.32,0.1]);
    plot(loc);grid on;



end 
function y = synth_onset(x, frameLen, frameHop, onsets)
% Synthesize each onset as a 1-frame long white noise signal, and add it to % the original audio signal.

% Input
% - x : input audio waveform
% - frameLen : frame length (in samples)
% - frameHop : frame hop size (in samples) - onsets : detected onsets (in frames)
% Output
% - y : output audio waveform which is the mixture of x and synthesized onset impulses.

    y = x';
    nframes = ceil(length(x) / frameHop);
    y = [y, zeros(1,frameHop*nframes - length(y))];% zero pad signal to have complete frames
    noise_window = wgn(1,frameLen,-5);
    for k = onsets
        y(1, k*frameLen: (k+1)*frameLen-1)
        y(1, k*frameLen: (k+1)*frameLen-1) = y(1, k*frameLen: (k+1)*frameLen-1) + noise_window;
    end
    
    y = y';

end
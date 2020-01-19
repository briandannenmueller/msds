function beats = beat_dp(wavData, frameLen, onsetStrength, tempoExpected, lambda)
% beat tracking by dynamic programming. %
% Input
% - wavData                      : the audio signal
% - frameLen                     : frame length (in samples)
% - onsetStrength                : onset strength in each audio frame
% - tempoExpected                : the expected tempo (in BPM)
% - lambda regularity objective  : tradeoff between the onset strength objective and beat
% Output
% - beats                        : the estimated beat sequence (in frame number)
    
    [~,fs] = audioread(wavData);
    
    dp = zeros(1,length(onsetStrength)); % dp sequence
    score_cum = onsetStrength; % score to the current point

    nframe = (fs/frameLen) / (tempoExpected/60); % the number of frames within a beat
    srange = round(-2*nframe):-round(nframe/2); % search range for previous beat
    p = -lambda*abs((log(srange/-nframe)).^2); % regularity penalty function

    % searching
    for i = max(-srange + 1):length(onsetStrength)
        timerange = i + srange;
        score_curr = score_cum(timerange) + p;
        [best_pre_score,loc] = max(score_curr);  % find the beat previous
        score_cum(i) = best_pre_score + onsetStrength(i);
        dp(i) = timerange(loc); % stores
    end
    
    % trace back to the best sequence
    [best_pre_score, beats] = max(score_cum); 
    while dp(beats(1)) > 0
        beats = [dp(beats(1)),beats];
    end


end
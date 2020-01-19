function [pitch, ap_pwr, rms_val] = myYinPoly(wavData, fs, tHop, tW, f0Min, f0Max, dp_th)
    % My YIN for polyphonic music piece.
    % The first estimate is performed at time 0 (i.e. the first integration
    % window is from the first sample to time tW, which means the window is
    % centered at tW/2) and the following estimates are spaced by tHop.
    %
    % Input
    %   - wavData       : input single-channel audio wave (a column vector)
    %   - fs            : sampling rate of wavData
    %   - tHop          : time interval between two adjacent estimates in second (default 0.01)
    %   - tW            : integration window size in second (default 0.025)
    %   - f0Min         : lowest possible F0 in Hz (default 40)
    %   - f0Max         : highest possible F0 in Hz (default 400)
    %   - dp_th         : the threshold of dips of d_prime(default 0.1)
    % Output
    %   - pitch         : estimated pitches (tow row vectors)
    %   - ap_pwr        : the corresponding d_prime, which is approximatedly
    %                       the aperiodic power over the total signal power. 
    %                       It can be used as the salience of this pitch
    %                       estimate.
    %   - rms_val       : the RMS value of the signal in the integration
    %                       window, which can also be used to determine if
    %                       there is a pitch in the signal or not.
    %
    % Author: Meiying(Melissa) Chen
    % Created: 09/21/2019
    % Last modified: 09/21/2019

    % default parameters for speech
    if nargin<7 dp_th=0.1; end
    if nargin<6 f0Max=400; end
    if nargin<5 f0Min=40; end
    if nargin<4 tW=0.025; end
    if nargin<3 tHop=0.01; end

    % Start your implementation here
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%  preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('start processing');
    x = wavData'; % turn column vector into row vector
    win = round(tW*fs); % window length 
    hop = round(tHop*fs); % hop length
    N = length(x);
    nframes = ceil((N - win) / hop); % number of frames
    % zero pad signal to have complete frames
    x = [x, zeros(1,hop*nframes + win - N)];
    x_frame = zeros(nframes, win);
    start = 1;
    % cut into frames
    disp('cutting wave into frames');
    for k = 1:nframes
        x_frame(k,:) = x(start:start + win - 1);
        start = start + hop;
    end
    
   
    
    
    %%%%%%%%%%%%%%%%%%%%  Step 2 difference function %%%%%%%%%%%%%%%%%%%
    tau_max = min(round(fs / f0Min), win); 
    d = zeros(nframes,tau_max);
    x_temp = [x_frame, zeros(nframes,win)];
    disp('calculating difference function');
    for tau = 0:tau_max-1
        for j = 1:win  
             d(:,tau+1) = d(:,tau+1) + (x_temp(:,j) - x_temp(:,j+tau)).^2;         
        end
    end
    

    %%%%%%%%%%%%%%%%  Step 3 cumulative mean normalisze  %%%%%%%%%%%%%%%%
    d_norm = zeros(nframes,tau_max);
    d_norm(:,1) = 1; % set d at lag 0 as 1
    
    rms_val = zeros(1, nframes); % calculate RMS
    ap_pwr = zeros(1, nframes); % calculate anverge d prime value
    
    disp('calculating cumulative mean normaliszed difference');
    for k = 1:nframes
        for tau = 1:tau_max-1
            d_norm(k,tau+1) = d(k,tau+1)/((1/tau) * sum(d(k,1:tau+1)));
        end
        rms_val(k) = rms(x_frame(k));
        ap_pwr(k) = mean(d_norm(k,:)); 
    end

    %%%%%%%%%%%%%%%%%%%%  Step 4 absolute threshold %%%%%%%%%%%%%%%%%%%%
    lag = zeros(2,nframes);
    tau_min = round(fs / f0Max); 
    % set the d value outside [tau_min, tau_max] into inf for calculation
    % convinience
    d_norm(:,1:tau_min-1) = inf;
    d_norm(:,tau_max+1:end) = inf;
    
    disp('applying absolute threshold');
    for k = 1:nframes
        l = find(d_norm(k,:) < dp_th,2);
        % if no value found or just found 1 value
        if(isempty(l) == 1)
            l = mink(d_norm(k,:),2);
        else
            if (length(l) == 1)
                l = [min(d_norm(k,:)), l];
            end
        end
        lag(1,k) = l(1);
        lag(2,k) = l(2);
    end

    
    
    pitch = zeros(2,nframes);
    for i = [1,2]
        %%%%%%%%%%%%%%%%%  Step 5 parabolic interpolation %%%%%%%%%%%%%%%%%

        % period, time, and pitch at every frame
        period = zeros(1,nframes);
        time = zeros(1,nframes);

        disp('aplying parabolic interpolation');
        for k = 1:nframes
            if(lag(i,k) > tau_min && lag(i,k) < tau_max)
                % calculate parabolic interpolation
                alpha = d_norm(k,lag(i,k)-1);
                beta = d_norm(k,lag(i,k));
                gamma = d_norm(k,lag(i,k)+1);
                peak = 0.5*(alpha - gamma)/(alpha - 2*beta + gamma);
                temp = lag(i,k) - 1 + peak;

                % limit period value in required range
                if(temp >= tau_min && temp <= tau_max) period(k) = temp;
                else period(k) = lag(i,k);end
            else
                period(k) = lag(i,k);
            end

            pitch(i,k) = fs/period(k); % calculate pitch
            time(k) = (k-1)*hop/fs;
        end
    end 
        
   
    disp('done')
    
  
    disp('plotting')  
    % plot rms and pitches along the whole wave
    figure(4);
    plot(time, pitch(1));grid on;
    xlabel('Time(s)');
    ylabel('Pitch(Hz)');
    figure(5);
    plot(time, ap_pwr);grid on;
    xlabel('Time(s)');
    ylabel('d prime');
    figure(6);
    plot(time, rms_val);grid on;
    xlabel('Time(s)');
    ylabel('RMS Value');

end






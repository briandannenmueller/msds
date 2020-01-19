clear, clc, close all;

tHop=0.01;
tW=0.0464; 
f0Min=40;
f0Max=600;
dp_th=0.1;

[wavData,fs] = audioread('violin_bassoon.wav');


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
        l = find(d_norm(k,:) < dp_th, 2);
        % if no value found or just found 1 value
        if(isempty(l) == 1)
            [~, l] = mink(d_norm(k,:),2);
        else
            if (length(l) == 1)
                [~,m] = min(d_norm(k,:));
                l = [m, l];
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
    % plo
    figure(1);
    plot(time, pitch(1,:));grid on;
    hold on;
    plot(time, pitch(2,:));grid on;
    xlabel('Time(s)');
    ylabel('Pitch(Hz)');



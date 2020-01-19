clear, clc, close all;

%% problem 1-a 

tHop=0.01;
tW=0.025;
f0Min=40;
f0Max=600;
dp_th=0.1;

[wavData,fs] = audioread('violin.wav');
[pitch, ap_pwr, rms_val] = myYin(wavData, fs, tHop, tW, f0Min, f0Max, dp_th);

%% problem 1-b

pEst = [0 1 0 1.01 2.3 2.92];
pGT = [0 0 0 1 2 3];
freqDevTh = 0.03;

[Acc, Pre, Rec] = evalSinglePitch(pEst, pGT, freqDevTh);


%% problem 1-c, 1-d

tHop=0.01;
tW=0.0464;
f0Min=40;
f0Max=2000;
dp_th=0.1;

[wavData,fs] = audioread('violin.wav');
[pitch, ap_pwr, rms_val] = myYin(wavData, fs, tHop, tW, f0Min, f0Max, dp_th);

% calculate accurancy, precision,  recall and f1 score
pitch_trim = pitch(1:length(pitch_gt)); % trim to same size
[Acc, Pre, Rec] = evalSinglePitch(pitch_trim, pitch_gt);
f1_score = 2*Rec / (Pre+Rec);


% grid search
rms_thres = (0.001:0.001:0.1);  
ap_pwr_thres = (0.1:0.05:1.2);
best_rms_thres = -1;
best_ap_pwr_thres = -1;

for r = rms_thres
    for a = ap_pwr_thres  

        % correction according to rms and ap_pwr value
        rms_cor = rms_val > r;
        ap_pwr_cor = ap_pwr < a;
        corr = rms_cor | ap_pwr_cor;
        pitch_cor = pitch .* corr;

        
        % calculate accurancy, precision,  recall and f1 score
        pitch_cor_trim = pitch_cor(1:length(pitch_gt)); % trim to same size
        [acc, pre, rec] = evalSinglePitch(pitch_cor_trim, pitch_gt);
        f1_temp = 2*rec / (pre+rec);
          

        if f1_temp > f1_score 
            best_ap_pwr_thres = a;
            best_rms_thres = r;
            f1_score = f1_temp;
            Acc = acc;
            Pre = pre;
            Rec = rec;
        end
    
    end
end


%% problem 1-e

tHop=0.01;
tW=0.0464; % keep this
f0Min=40;
f0Max=1000;
dp_th=0.01;

[wavData,fs] = audioread('violin_noise.wav');
[pitch, ap_pwr, rms_val] = myYin(wavData, fs, tHop, tW, f0Min, f0Max, dp_th);

% calculate accurancy, precision,  recall and f1 score
pitch_trim = pitch(1:length(pitch_gt)); % trim to same size
[Acc, Pre, Rec] = evalSinglePitch(pitch_trim, pitch_gt);
f1_score = 2*Rec / (Pre+Rec);

% grid search
rms_thres = (0.001:0.001:0.1);  
ap_pwr_thres = (0.1:0.05:1.2);
best_rms_thres = -1;
best_ap_pwr_thres = -1;


for r = rms_thres
    for a = ap_pwr_thres  

        % correction according to rms and ap_pwr value
        rms_cor = rms_val > r;
        ap_pwr_cor = ap_pwr < a;
        corr = rms_cor | ap_pwr_cor;
        pitch_cor = pitch .* corr;

        
        % calculate accurancy, precision,  recall and f1 score
        pitch_cor_trim = pitch_cor(1:length(pitch_gt)); % trim to same size
        [acc, pre, rec] = evalSinglePitch(pitch_cor_trim, pitch_gt);
        f1_temp = 2*rec / (pre+rec);
          

        if f1_temp > f1_score
            best_rms_thres = r;
            best_ap_pwr_thres = a;
            f1_score = f1_temp;
            Acc = acc;
            Pre = pre;
            Rec = rec;
        end
    end
end


%% problem 1-f
[wavData,fs] = audioread('violin_noise.wav');
w = hamming(512);
noverlap = 256;
nfft = 512;
spectrogram(wavData, w, noverlap, nfft, fs,'yaxis');


%% problem 2-a
tHop=0.01;
tW=0.0464;
f0Min=390;
f0Max=530;
dp_th=0.1;

[wavData,fs] = audioread('violin_bassoon.wav');
[pitch, ap_pwr, rms_val] = myYin(wavData, fs, tHop, tW, f0Min, f0Max, dp_th);


%% problem 2-b
tHop=0.01;
tW=0.0464; 
f0Min=40;
f0Max=600;
dp_th=0.1;

[wavData,fs] = audioread('violin_noise.wav');
[pitch, ap_pwr, rms_val] = myYinPoly(wavData, fs, tHop, tW, f0Min, f0Max, dp_th);




















        
        
        
        
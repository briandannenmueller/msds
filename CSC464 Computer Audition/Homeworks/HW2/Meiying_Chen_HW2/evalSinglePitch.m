function [Acc, Pre, Rec] = evalSinglePitch(pEst, pGT, freqDevTh)
% Evaluate single pitch detection results
%
% Input
%   - pEst      : the pitch estimate in Hz (a row vector)
%   - pGT       : the ground-truth pitch in Hz (a row vector)
%   - freqDevTh : the maximal percentage of Hz that an estimated pitch can be
%                   deviated from the ground-truth pitch such that the
%                   estimated pitch is still judged as a correct estimate.
%                   (Default: 0.03)
% Output
%   - Acc       : estimation accuracy: nC/(nG+nE-nC), i.e. among all
%                   estimated and ground-truth non-zero pitches, what
%                   percentage are correct. 
%   - Pre       : estimation precision: nC/nE, i.e. among all estimated
%                   non-zero pitches, what percentage are correct.
%   - Rec       : estimation recall: nC/nG, i.e. among all ground-truth
%                   non-zero pitches, what percentage are correctly
%                   estimated. 
%
% Author: Meiying(Melissa) Chen
% Created: 09/16/2019
% Last modified: 09/19/2019

if nargin<3 freqDevTh=0.03; end

% Start your implementation here

% decide if the estimation is in the required range
upper = pGT * (1 + freqDevTh);
lower = pGT * (1 - freqDevTh);
iscorrect = pEst >= lower & pEst <= upper; % 1 for correctly estimated, 0 for not correctly estimated 


% Accurancy
% locations where both estimation and ground truth are not 0
both_non0 = pGT .* pEst; 
% locations where both estimation and ground truth are not 0 && the estimation is correct
cor_both_non0 = both_non0 .* iscorrect; 
Acc = sum(cor_both_non0 ~= 0) / sum(both_non0 ~= 0);


% Precision
cor_est_non0 = pEst .* iscorrect; % locations where estimations are not 0 and correct
Pre = sum(cor_est_non0 ~= 0) / sum(pEst ~= 0);


% Recall
cor_gt_non0 = pGT .* iscorrect;
Rec = sum(cor_gt_non0 ~= 0) / sum(pGT ~= 0);


end

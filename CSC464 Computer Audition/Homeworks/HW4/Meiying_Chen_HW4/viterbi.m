function [ v , prev ] = viterbi( transMat, loglikeMat, prev ,T ,initP)
%%%% Reference: https://github.com/hzx829/Computer-Audition/edit/master/HMM_model_Pitch_Contour_Estimation/viterbi.m 
    nState = size(transMat,1);
    prev_j = zeros(1,nState);

    if T == 1
        v = zeros(1,nState);
        for state_c = 1:nState  %current state
            max_prevj = log(0);
            for state_p = 1:nState  % previous state
                log_p = log(initP(state_p))+ log( transMat(state_p,state_c)) ;
                if log_p > max_prevj
                    max_prevj = log_p;
                    argmax_prevj = state_p;
                end

            end
            prev_j(state_c) = argmax_prevj;
            v(state_c) = max_prevj + loglikeMat(state_c,T);
        end
        [~,argmax_v] = max(v);
        prev(T) = prev_j(argmax_v);


    end

    if T > 1
        [max_v_before,prev] = viterbi(transMat,loglikeMat,prev(1:T-1),T-1,initP);
        for state_c = 1:nState  %current state
            max_prevj = log(0);
            for state_p = 1:nState  % previous state

                log_p = max_v_before(state_p)+log(transMat(state_p,state_c));
                if log_p > max_prevj
                    max_prevj = log_p;
                    argmax_prevj = state_p;
                end

            end
            prev_j(state_c) = argmax_prevj;
            v(state_c) = max_prevj + loglikeMat(state_c,T);
        end
        [~,argmax_v] = max(v);
        prev(T) = prev_j(argmax_v);



    end

end


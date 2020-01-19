%{
s_paths = ["/Users/dotdot/Data/test/Sources/60.wav" "/Users/dotdot/Data/test/Sources/63.wav" "/Users/dotdot/Data/test/Sources/66.wav" "/Users/dotdot/Data/test/Sources/65.wav" "/Users/dotdot/Data/test/Sources/59.wav" "/Users/dotdot/Data/test/Sources/55.wav" "/Users/dotdot/Data/test/Sources/56.wav" "/Users/dotdot/Data/test/Sources/52.wav" "/Users/dotdot/Data/test/Estimated/53.wav" "/Users/dotdot/Data/test/Sources/51.wav"];
se_paths = ["/Users/dotdot/Data/test/Estimated/60.wav" "/Users/dotdot/Data/test/Estimated/63.wav" "/Users/dotdot/Data/test/Estimated/66.wav" "/Users/dotdot/Data/test/Estimated/65.wav" "/Users/dotdot/Data/test/Estimated/59.wav" "/Users/dotdot/Data/test/Estimated/55.wav" "/Users/dotdot/Data/test/Estimated/56.wav" "/Users/dotdot/Data/test/Estimated/52.wav" "/Users/dotdot/Data/test/Estimated/53.wav" "/Users/dotdot/Data/test/Estimated/51.wav"];

SDRs = [];

for i  = (1:1:length(s_paths))
    [se, ~] = audioread(se_paths(i));
    [s, ~] = audioread(s_paths(i));
    disp(se_paths(i))
    disp(s_paths(i))
    len = min(length(s), length(se));
    se = se(1:len);
    s = s(1:len);
    [SDR,SIR,SAR,perm] = bss_eval_sources(se.',s.');
    SDRs(i) = SDR;
end

sum = 0
for i  = (1:1:length(s_paths))
    sum = sum + SDRs(i)
end
ave = sum / length(s_paths)

%}
    clear;
    [se, ~] = audioread("estimated.wav");
    [s, ~] = audioread("true.wav");
    len = min(length(s), length(se));
    se = se(1:len);
    s = s(1:len);
    [SDR,SIR,SAR,perm] = bss_eval_sources(se.',s.');
   
    


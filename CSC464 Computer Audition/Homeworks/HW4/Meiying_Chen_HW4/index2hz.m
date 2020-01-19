function [hz] = index2hz(index)
    midi = (index - 2) * 0.5 + 32.25;
    hz = 2.^((midi - 69)./ 12).* 440;
end
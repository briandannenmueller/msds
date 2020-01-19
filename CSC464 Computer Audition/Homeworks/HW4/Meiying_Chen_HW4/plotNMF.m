function []  = plotNMF(V, V_res, x, x_res, W, H, r, fs, t)

    % plot the original spectrogram
    figure(1);
    imagesc(t,fs,20*log10(abs(V)));
    xlabel('time (second)');
    ylabel('frequency (Hz)');
    set(gca,'YDir','normal');
    colorbar

    % plot the restored spectrogram
    figure(2);
    imagesc(t,fs,20*log10(abs(V_res)));
    xlabel('time (second)');
    ylabel('frequency (Hz)');
    set(gca,'YDir','normal');
    colorbar
    
    % plot the original wave
    figure(3);
    plot(x);
    ylabel('amplitude');
    
    % plot the restored wave
    figure(4);
    plot(x_res);
    ylabel('amplitude');
    
    % plot cols of W and rows of H
    for i = 1 : r
        figure(5);
        subplot(1,r,i);
        plot(abs(W(:,i)));
        axis([0,500,0,0.5]);
        xlabel('frequency(Hz)');
        ylabel('amplitude');
        title(['W',num2str(i)])
        figure(6);
        subplot(r,1,i);
        plot(abs(H(i,:)));
        title(['H',num2str(i)])
        xlabel('frame');
        ylabel('activation');
    end

    

end
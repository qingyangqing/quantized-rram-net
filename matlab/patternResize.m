function outPatterns=patternResize(inPatterns, size, imresolution)
    % data format from double to uint8
    inPatterns=uint8(inPatterns*255);
    len=length(inPatterns);
    outPatterns=zeros(size*size,len);
    for i=1:len
        temp1=inPatterns(:,i);
        temp2=reshape(temp1,28,28);
        % resize 
        steps=256/(2^imresolution);
        step_size=1/(2^imresolution);
        temp3=double(imresize(temp2, [size size])/steps)*step_size;
        outPatterns(:,i)=temp3(:);
    end
end
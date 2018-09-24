function outLabel=labelResize(inLabel)
    len=length(inLabel);
    outLabel=zeros(10,len);
    for i=1:len
        outLabel(inLabel(i)+1,i)=1;
    end
end
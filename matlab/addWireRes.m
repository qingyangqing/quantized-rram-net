function [output]=addWireRes(input)
    temp1=abs(input');
    [r,c]=size(temp1);
    temp1=1./temp1;
    for i=1:r
        for j=1:c
            temp1(i,j)=temp1(i,j)+5.6533E-3*j+2.256E-3*i;
        end
    end
    temp1=1./temp1;
    temp2=(input>0)-(input<0);
    output=(temp1').*temp2;
end

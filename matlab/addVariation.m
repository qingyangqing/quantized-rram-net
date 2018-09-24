function [output]=addVariation(input)
    [r,c]=size(input);
    temp1=ones(r,c);
    temp2=temp1+0.05*rands(r,c);
    output=input.*temp2;
end
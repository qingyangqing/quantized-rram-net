function [W_q]=WeightQuantize(W, gg)
    W_q=zeros(size(W));
    len=length(gg);
    gg_inter=(gg(1:len-1)+gg(2:len))/2;
    % quantize W
    len=length(gg_inter);
    W_q=W_q+gg(1)*(W<gg_inter(1));
    for i=2:len
        W_q=W_q+gg(i)*((W>=gg_inter(i-1)) .* (W<gg_inter(i)));
    end
    W_q=W_q+gg(len+1)*(W>=gg_inter(len));
end
function [gg, gg_mat]=ggGenerate(g)
    len=length(g);
    gg_mat=zeros(len);
    for i=1:len
        gg_mat(i,:)=g(i)-g;
    end
    gg=unique(sort(gg_mat(:)));
end
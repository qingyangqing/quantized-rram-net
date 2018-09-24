%% scale and mapped to resistance 
% gg combination and gg_matrix
rram_level=3;
g=1./linspace(10,500,rram_level);
[gg, gg_mat]=ggGenerate(g);
accuracy_vs_scale=[];
for scale1=1:100
    W1_new=W1/scale1;
    [W1_q]=WeightQuantize(W1_new, gg);
    N1=W1_q*mnist_test;
    A1=poslin(N1);
    for scale2=1:100
        W2_new=W2/scale2;
        [W2_q]=WeightQuantize(W2_new, gg);
        N2=W2_q*A1;
        A2=softmax(N2);
        [~,k]=max(A2);
        accuracy_vs_scale=[accuracy_vs_scale, length(find(j==k))/length(mnist_test)];
    end
end
%% find optimal W1_q and W2_q
position=find(accuracy_vs_scale==max(accuracy_vs_scale));
scale1=fix(position/100)+1;
scale2=mod(position,100);
if scale2(1)==0  
    scale2(1)=100; 
    scale1(1)=scale1(1)-1;
end
W1_new=W1/scale1(1);
[W1_q]=WeightQuantize(W1_new, gg);
W2_new=W2/scale2(1);
[W2_q]=WeightQuantize(W2_new, gg);
% test again after choosing optimal scaling factors
N1=W1_q*mnist_test;
A1=poslin(N1);
N2=W2_q*A1;
A2=softmax(N2);
[~,k]=max(A2);
fprintf('Accuracy after mapping: %1f \n', length(find(j==k))/length(mnist_test));

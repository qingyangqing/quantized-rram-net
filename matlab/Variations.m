% variations 
[W1_va]=addVariation(W1_q);
[W2_va]=addVariation(W2_q);
[mnist_test_va]=addVariation(mnist_test);
N1=W1_va*mnist_test_va;
A1=TransferFunctionInter(N1);
N2=W2_va*A1;
A2=TransferFunction(N2);
[~,k]=max(A2);
fprintf('Error Rate with variations: %1f \n', 1-length(find(j==k))/NumTotal);

% add wire resistance
[W1_wres]=addWireRes(W1_q);
[W2_wres]=addWireRes(W2_q);
N1=W1_wres*mnist_test;
A1=TransferFunctionInter(N1);
N2=W2_wres*A1;
A2=TransferFunction(N2);
[~,k]=max(A2);
fprintf('Error Rate with wire resistance: %1f \n', 1-length(find(j==k))/NumTotal);
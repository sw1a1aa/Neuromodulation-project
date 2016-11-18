tau_VIP=3.5; tau_SST=3.5; tau_PV=3.5; tau_Pyr=3.5;
I_VIP=70; I_SST=50; I_PV=70; I_Pyr=150;
r_VIP=37; r_SST=21; r_PV=15; r_Pyr=27;

sigmoid = @(x,b,T)(80./(1+exp(-b*(x-T))));

tf_VIP = sigmoid(I_VIP - 0.16*r_SST, 0.09, 70);
tf_SST = sigmoid(I_SST - 0.16*r_VIP-0.16*r_PV, 0.09, 70);
tf_PV = sigmoid(I_PV - 0.16*r_SST-0.1*r_PV, 0.09, 70);
tf_Pyr = sigmoid(I_Pyr - 0.3*r_PV - 0.35*r_SST, 0.05, 150);

VIP=1/tau_VIP*[-1, -0.16*tf_VIP*(1-tf_VIP), 0, 0];
SST=1/tau_SST*[-0.16*tf_SST*(1-tf_SST), -1, -0.16*tf_SST*(1-tf_SST), 0];
PV=1/tau_PV*[0, -0.16*tf_PV*(1-tf_PV), -1-0.1*tf_PV*(1-tf_PV), 0];
Pyr=1/tau_Pyr*[-1, -0.35*tf_Pyr*(1-tf_Pyr), -0.3*tf_Pyr*(1-tf_Pyr), -1];
A=[VIP;SST;PV;Pyr];

[evecs,evals]=eig(A)
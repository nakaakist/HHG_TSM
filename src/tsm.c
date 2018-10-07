#include <math.h>
#include <stdlib.h>

double ADK(double E, double Cnl, double Glm, double Ip, double F0, double n, int m){
  return Cnl*Cnl*Glm*Ip*pow(2*F0/fabs(E), 2*n-abs(m)-1)*exp(-2*F0/(3*fabs(E)));
}

void tsm(int N, void* T_v, void* E_v, int gas, void* X_v){
  double* T = (double*)T_v;
  double* E = (double*)E_v;
  double* X = (double*)X_v;
  double W[N];
  double A[N];
  double Aint[N];
  double A2int[N];
  double ip;

  double dt = (T[N-1]-T[0])/N;

  //gas: 0: helium, 1:neon, 2:argon, 3:krypton
  switch(gas){
  case 0:
    for(int i=0; i<N; i++){
      W[i] = ADK(E[i], 2.06295, 1, 24.587/27.21, 2.42946, 0.74387, 0);
      ip = 24.587/27.21;
    }
    break;
  case 1:
    for(int i=0; i<N; i++){
      W[i] = ADK(E[i], 2.05999, 3, 21.56/27.21, 1.99547, 0.7943, 0);
      ip = 21.56/27.21;
    }
    break;
  case 2:
    for(int i=0; i<N; i++){
      W[i] = ADK(E[i], 2.02870, 3, 15.85/27.21, 1.24665, 0.92915, 0);
      ip = 15.85/27.21;
    }
    break;
  case 3:
    for(int i=0; i<N; i++){
      W[i] = ADK(E[i], 2.00636, 3, 14.35/27.21, 1.04375, 0.98583, 0);
      ip = 14.35/27.21;
    }
    break;
  }

  A[0] = 0;
  for(int i=1; i<N; i++){
    A[i] = A[i-1]+dt*E[i-1];
  }

  Aint[0] = 0;
  A2int[0] = 0;
  for(int i=1; i<N; i++){
    Aint[i] = Aint[i-1]+dt*A[i-1];
    A2int[i] = A2int[i-1]+dt*A[i-1]*A[i-1];
  }

  double x_diff1;
  double x_diff2;
  int icnt;
  double t, e, a, aint, a2int, w, pst, sst;
  double x;
  int num_traj = 10;
  int I_cross[num_traj];
  double I_resid[num_traj];

  for(int i=0; i<N; i++){

    icnt = 0;
    x_diff1 = 0;
    for(int j=10; j<i; j++){
      x_diff2 = (Aint[i]-Aint[i-j])/(T[i]-T[i-j])-A[i-j];
      if((x_diff1*x_diff2) < 0){
	I_cross[icnt] = i-j;
	I_resid[icnt] = fabs(x_diff2/(x_diff2-x_diff1));
	icnt++;
	if(icnt >= num_traj){
	  break;
	}
      }
      x_diff1 = x_diff2;
    }

    x = 0;
    for(int j=0; j<icnt; j++){
      t = T[I_cross[j]]+(T[I_cross[j]+1]-T[I_cross[j]])*I_resid[j];
      e = E[I_cross[j]]+(E[I_cross[j]+1]-E[I_cross[j]])*I_resid[j];
      a = A[I_cross[j]]+(A[I_cross[j]+1]-A[I_cross[j]])*I_resid[j];
      aint = Aint[I_cross[j]]+(Aint[I_cross[j]+1]-Aint[I_cross[j]])*I_resid[j];
      a2int = A2int[I_cross[j]]+(A2int[I_cross[j]+1]-A2int[I_cross[j]])*I_resid[j];
      w = W[I_cross[j]]+(W[I_cross[j]+1]-W[I_cross[j]])*I_resid[j];
      pst = (Aint[i]-aint)/(T[i]-t);
      sst = (0.5*pst*pst+ip)*(T[i]-t) - pst*(Aint[i]-aint) + (A2int[i]-a2int)*0.5;
      x += pow(1/(T[i]-t), 1.5)*
	   cos(sst)*
	   (pst-A[i])/pow(pow(pst-A[i], 2)+2*ip, 3)*
	   sqrt(w)/fabs(e);
    }
    X[i] = x;
  }
}

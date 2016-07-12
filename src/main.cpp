#include <stdio.h>
#include <math.h>

int main(){
  double fs = 1000/24.18884;
  double ev = 1/27.211;
  double mv_per_cm  = 1/(5.142206*pow(10, 3));
  double angstrom = 1/0.529177;
  double tmax = 20*fs;
  double emax = 290*mv_per_cm;
  double efwhm = 10*fs;
  double etau = 5.33*fs;
  double ip = 14*ev;
  double Cnl = 2.05999;
  double Glm = 3;
  double Ip = 21.56*ev;
  double F0 = 1.99547;
  double n_star = 0.7943;
  double m_star = 0;

  int N = 5000;

  double dt = 2*tmax/N;

  double T[N];
  double E[N];
  double A[N];
  double Aint[N];
  double A2int[N];
  double W[N];
  double X[N];

  for(int i=0; i<N; i++){
    T[i] = -tmax+dt*i;
    E[i] = emax*cos(2*M_PI*T[i]/etau)*exp(-pow(T[i]/efwhm, 2));
    W[i] = Cnl*Cnl*Glm*Ip*pow((2*F0/fabs(E[i])), 2*n_star-fabs(m_star)-1)*exp(-2*F0/(3*fabs(E[i])));
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
    printf("%e\n", x);
  }

  return 0;
}

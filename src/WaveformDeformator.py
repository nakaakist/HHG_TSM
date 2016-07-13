from units import *
import numpy as np
import pandas as pd
from scipy import linalg

class WaveformDeformator:

  def __init__(self, rmax=400*um, rnum=300, r0_beam=100*um, zrange=6000*um,
               f=4000*um, znum=400, tmax=20*fs_si, tnum=5000, l0=1.6*um,
               emax=280*mv_per_cm, e_fwhm=10*fs_si, cep=0*np.pi, wmax=2*np.pi*2*10**15):
    self.rmax = rmax
    self.rnum = rnum
    self.r0_beam = r0_beam
    self.zrange = zrange
    self.f = f
    self.znum = znum
    self.tmax = tmax
    self.tnum = tnum
    self.l0 = l0
    self.emax = emax
    self.e_fwhm = e_fwhm
    self.cep = cep
    self.wmax = wmax

    self.dr = rmax/rnum
    self.dt = 2*tmax/tnum
    self.dt_atomic = 2*tmax/tnum/fs_si*fs
    self.dw = 2*np.pi/self.dt/tnum
    self.dz = zrange/znum

    self.w0 = 2*np.pi*c/l0
    self.w0_atomic = 1.24/(2*np.pi*c/self.w0/um)*ev

    self.Z = np.linspace(0, zrange, znum)
    self.R = np.linspace(0, rmax, rnum)
    self.T = np.linspace(-tmax, tmax, tnum)
    self.Omega = np.linspace(0, 2*np.pi/self.dt, tnum)
    self.Ev = 1.24/(2*np.pi*c)*self.Omega*um
    self.Ri, self.Ti = np.meshgrid(self.R, self.T)
    self.Ri, self.Omegai = np.meshgrid(self.R, self.Omega)
    self.__calc_initial_field()

  def propagate(self, nonlinear=True, gas='Ar', p_g=1, neutral_disp=False, wmax=2*np.pi*2*10**15):
    l = 1
    u = 1
    ab = j*np.zeros((l+u+1, self.rnum-1))

    self.Etz = []
    self.Erz = []
    self.N_kerrz =[]
    self.N_electronz = []
    self.Pz = []
    I1 = np.arange(1, self.rnum-2)
    I2 = np.clip(np.arange(0, self.rnum-1), 1, self.rnum)
    Ew = np.fft.fft(self.E0, axis=0)
    self.Sw = 0*Ew
    Reduced_omega = self.Omega[self.Omega < wmax]
    for iz, z in enumerate(self.Z):

      #source term
      if nonlinear:
        self.__calc_source_term(Ew, gas, neutral_disp, p_g)
      else:
        self.Sw = 0*Ew

      #propagation
      for i, omega in enumerate(Reduced_omega):
        if omega == 0:
          continue
        C = 0.5*c/j/omega*self.dz/self.dr**2
        D = 0.5*c/j/omega

        ab[u+0-0, 0] = 0.5*C
        ab[u+0-1, 1] = 1-0.5*C
        ab[u+1, I1-1] = -0.5*C*(1-0.5/I1)
        ab[u+0, I1] = 1+C
        ab[u-1, I1+1] = -0.5*C*(1+0.5/I1)
        ab[u+1, self.rnum-3] = -0.5*C*(1-0.5/(self.rnum-2))
        ab[u+0, self.rnum-2] = 1+C

        B = 0.5*C*(1-0.5/I2)*Ew[i, I2-1]+(1-C)*Ew[i, I2]+0.5*C*(1+0.5/I2)*Ew[i, I2+1]-D*self.dz*self.Sw[i, I2]
        B[0] = -0.5*C*Ew[i, 0]+(1+0.5*C)*Ew[i, 1]-D*self.dz*self.Sw[i, 0]
        X = linalg.solve_banded((1, 1), ab, B)
        Ew[i, :] = np.append(X, 0)
      E = np.fft.ifft(Ew, axis=0)
      self.Etz.append(E[:, 0])
      self.Erz.append((np.abs(E)**2).sum(axis=0))

    self.Etz = np.array(self.Etz)
    self.Erz = np.array(self.Erz)
    self.E = np.array(E)
    self.N_kerrz = np.array(self.N_kerrz)
    self.N_electronz = np.array(self.N_electronz)
    self.Pz = np.array(self.Pz)
    if not nonlinear:
      self.Etz_linear = np.array(self.Etz)
      self.Erz_linear = np.array(self.Erz)
      self.E_linear = np.array(E)

  def evaluate_convergence(self):
    I = []
    for i, z in enumerate(self.Z):
      I.append((self.Erz[i, :]*self.R).sum())
    plt.plot(self.Z/um, I)
    plt.xlabel('Z ($¥mu$m)')
    plt.ylabel('Total pulse energy')
    plt.ylim(0, max(I)*1.1)

  def plot_Etz(self, zmax=False):
    Ti, Zi = np.meshgrid(self.T, self.Z)
    if not zmax:
      plt.contourf(Ti/fs_si, Zi/um, np.abs(self.Etz)**2/w_per_cm2, 50)
    else:
      plt.contourf(Ti/fs_si, Zi/um, np.abs(self.Etz)**2/w_per_cm2, np.linspace(0, zmax, 50))
    plt.colorbar()
    plt.xlabel('Time (fs)')
    plt.ylabel('Z ($\mu$m)')

  def plot_Erz(self):
    Ri, Zi = np.meshgrid(self.R, self.Z)
    plt.contourf(Ri/um, Zi/um, self.Erz/w_per_cm2, 50)
    plt.colorbar()
    plt.xlabel('R ($\mu$m)')
    plt.ylabel('Z ($\mu$m)')

  def compare_waveform(self):
    plt.plot(self.T/fs_si, (self.Etz_linear[-1, :]).real, label='linear')
    plt.plot(self.T/fs_si, (self.Etz[-1, :]).real, label='nonlinear')
    plt.xlabel('Time (fs)')
    plt.ylabel('Electric field (arb. unit)')
    plt.legend(loc='best')

  def compare_spectrum(self, maxev=2):
    Spec_linear = np.abs(np.fft.fft(self.Etz_linear[-1, :]))**2
    Spec_nonlinear = np.abs(np.fft.fft(self.Etz[-1, :]))**2
    plt.plot(self.Ev, Spec_linear/max(Spec_linear), label='linear')
    plt.plot(self.Ev, Spec_nonlinear/max(Spec_nonlinear), label='nonlinear')
    plt.xlim(0, maxev)
    plt.xlabel('Photon energy (ev)')
    plt.ylabel('Intensity (arb. unit)')
    plt.legend(loc='best')

  def __calc_initial_field(self):
    Ew0 = np.fft.fft(np.exp(-(self.T/0.8493/self.e_fwhm)**2+j*(self.w0*self.T+self.cep)))
    E0 = j*0*self.Ri
    Reduced_omega = self.Omega[self.Omega < self.wmax]
    for i, omega in enumerate(Reduced_omega):
      if omega == 0:
        continue
      l = 2*np.pi*c/omega
      zr = np.pi*self.r0_beam**2/l
      r_beam = self.r0_beam*np.sqrt(1+(self.f/zr)**2)
      rz = self.f+zr**2/self.f
      k = 2*np.pi/l
      E0 += np.exp(-(self.Ri/r_beam)**2+0.5*j*k*(self.Ri**2/rz)-j*np.arctan(self.f/zr))*Ew0[i]*np.exp(j*omega*(self.Ti-self.tmax))
    E0 = np.array(E0)
    self.E0 = E0/(np.abs(E0).max())*self.emax

  def __calc_source_term(self, Ew, gas, neutral_disp, p_g):
    E = np.fft.ifft(Ew, axis=0)
    for i, r in enumerate(self.R):
      if gas == 'Ne':
        P = 1-np.exp(-self.__ADK(E[:, i].real, 2.05999, 3, 21.56*ev, 1.99547, 0.7943, 0).cumsum()*self.dt_atomic) #Ne
        N_gas = self.__n_gas(p_g, P, 16.3*ev, self.w0_atomic)-self.__n_gas(p_g, 1, 8*ev, self.w0_atomic) #Ne
        N_kerr = self.__n_kerr(p_g, E[:, i], 1.31*10**(-24)) #Ne
      elif gas == 'Ar':
        P = 1-np.exp(-self.__ADK(E[:, i].real, 2.02870, 3, 15.85*ev, 1.24665, 0.92915, 0).cumsum()*self.dt_atomic) #Ar
        N_gas = self.__n_gas(p_g, P, 8*ev, self.w0_atomic) #Ar
        N_kerr = self.__n_kerr(p_g, E[:, i], 9.8*10**(-24)) #Ar
      elif gas == 'Kr':
        P = 1-np.exp(-self.__ADK(E[:, i].real, 2.00636, 3, 14.35*ev, 1.04375, 0.98583, 0).cumsum()*self.dt_atomic) #Kr
        N_gas = self.__n_gas(p_g, P, 6.5*ev, self.w0_atomic) #Kr
        N_kerr = self.__n_kerr(p_g, E[:, i], 27.8*10**(-24)) #Kr
      W_plasma_square = self.__w_plasma_square(p_g, P)
      tau_i = int((np.pi/self.w0)/self.dt)
      W_plasma_square = pd.rolling_mean(W_plasma_square, tau_i, center=True)
      W_plasma_square[-int(tau_i/2):] = np.nanmin(W_plasma_square)
      W_plasma_square[:int(tau_i/2)] = 0
      W_plasma_square = W_plasma_square/(1+np.exp((self.T-self.T.max()+2*np.pi/self.w0)*self.w0))
      if not neutral_disp:
        N_gas -= N_gas.max()
      self.Sw[:, i] = -2*np.fft.fft((N_gas+N_kerr)*E[:, i]/c**2)*self.Omega**2+np.fft.fft(W_plasma_square*E[:, i]/c**2)
      if r == 0:
        self.N_kerrz.append(N_kerr)
        self.N_electronz.append(0.5*W_plasma_square/self.w0**2)
        self.Pz.append(P)

  def __ADK(self, E, Cnl, Glm, Ip, F0, n, m):
    return Cnl**2*Glm*Ip*(2*F0/np.abs(E))**(2*n-np.abs(m)-1)*np.exp(-2*F0/(3*np.abs(E)))

  def __n_gas(self, p_g, p, Ie, w):
    return 2*np.pi*(2.5*10**(-5)*(1/angstrom)**3)*p_g*((1-p)/(Ie**2-w**2))

  def __w_plasma_square(self, p_g, p):
    return 4*np.pi*(2.5*10**(-5)*(1/angstrom)**3)*p_g*p*(10**(15)*fs)**2

  def __n_kerr(self, p_g, E, n2):
    return p_g*np.abs(E)**2*(3.51*10**(20))*n2

import os
from units import *
import numpy as np
import pandas as pd
from scipy import linalg, interpolate
import ctypes
import multiprocessing as mp

tsmlib = ctypes.cdll.LoadLibrary('%s/tsm.so' % os.path.dirname(__file__))
tsm = tsmlib.tsm

class WaveformDeformator:

  def __init__(self, rmax=400*um, rnum=300, r0_beam=100*um, zrange=6000*um,
               f=4000*um, znum=400, tmax=25*fs_si, tnum=1000, l0=1.6*um,
               emax=280*mv_per_cm, e_fwhm=10*fs_si, cep=0*np.pi, wmax=2*np.pi*2*10**15, tnum_hhg=5000):
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

    self.tnum_hhg = tnum_hhg
    self.T_hhg = np.linspace(-tmax*fs/fs_si, tmax*fs/fs_si, tnum_hhg)
    self.dt_hhg = self.dt*tnum/tnum_hhg
    self.Omega_hhg = np.linspace(0, 2*np.pi/self.dt_hhg, tnum_hhg)
    self.Ev_hhg = 1.24/(2*np.pi*c)*self.Omega_hhg*um
    self.gas_c_dict = {'Ne': 1, 'Ar': 2, 'Kr': 3}
    self.gas_ip_dict = {'Ne': 21.56*ev, 'Ar': 15.85*ev, 'Kr': 14.35*ev}

  def propagate(self, nonlinear=True, gas='Ar', p_g=1, neutral_disp=True, wmax=2*np.pi*2*10**15, hhg=True):

    self.__calc_initial_field()
    l = 1
    u = 1
    ab = j*np.zeros((l+u+1, self.rnum-1))

    if hhg:
      ref_data = pd.read_csv('%s/constants/%s_refractive_indices.dat' % (os.path.dirname(__file__), gas), skiprows=10, delimiter='\t', names=['energy', 'd', 'b'])
      f_beta = interpolate.interp1d(ref_data.energy, ref_data.b, kind='cubic')
      Beta_hhg = f_beta(np.clip(self.Ev_hhg, 40, 500))

    self.Etz = []
    self.Erz = []
    self.Pz = []
    self.X_hhg = j*np.zeros(self.tnum_hhg)
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

      if hhg:
        E_hhg = np.fft.ifft(np.append(Ew[:, 0], np.zeros(self.tnum_hhg-self.tnum)))
        E_hhg = (E_hhg.real/max(np.abs(E_hhg))*max(np.abs(E[:, 0]))).astype(np.float)
        X = np.zeros(self.tnum_hhg)
        tsm(self.tnum_hhg,
            ctypes.c_void_p(self.T_hhg.ctypes.data),
            ctypes.c_void_p(E_hhg.ctypes.data),
            self.gas_c_dict[gas],
            ctypes.c_void_p(X.ctypes.data))
        X_spec = np.append(np.fft.fft(X)[:int(self.tnum_hhg/2)], j*np.zeros(int(self.tnum_hhg/2)))
        X_spec *= np.exp(-Beta_hhg*2*np.pi/(1240/self.Ev_hhg*10**(-9))*(self.zrange-z)*p_g)*p_g*(self.zrange/self.znum)
        self.X_hhg += np.fft.ifft(X_spec)

    self.Etz = np.array(self.Etz)
    self.Erz = np.array(self.Erz)
    self.E = np.array(E)
    self.Pz = np.array(self.Pz)
    if not nonlinear:
      emax_correction = self.emax/(np.abs(self.Etz).max())
      self.E0 = self.E0*emax_correction
      self.Etz_linear = np.array(self.Etz)*emax_correction
      self.Erz_linear = np.array(self.Erz)*emax_correction**2
      self.E_linear = np.array(E)*emax_correction


  def __calc_initial_field(self):
    Ew0 = np.fft.fft(np.exp(-(self.T/0.8493/self.e_fwhm)**2+j*(self.w0*self.T+self.cep)))
    E0 = j*0*self.Ri
    Reduced_omega = self.Omega[self.Omega < self.wmax]
    for i, omega in enumerate(Reduced_omega):
      if omega == 0:
        continue
      l = 2*np.pi*c/omega
      zr = np.pi*self.r0_beam**2/l
      self.r_beam = self.r0_beam*np.sqrt(1+(self.f/zr)**2)
      rz = self.f+zr**2/self.f
      k = 2*np.pi/l
      E0 += np.exp(-(self.Ri/self.r_beam)**2+0.5*j*k*(self.Ri**2/rz)-j*np.arctan(self.f/zr))*Ew0[i]*np.exp(j*omega*(self.Ti-self.tmax))
    E0 = np.array(E0)
    self.E0 = E0/(np.abs(E0).max())*self.emax/np.sqrt(1+(self.f/(np.pi*self.r0_beam**2/self.l0))**2)

  def __calc_source_term(self, Ew, gas, neutral_disp, p_g):
    E = np.fft.ifft(Ew, axis=0)
    for i, r in enumerate(self.R):
      if gas == 'Ne':
        P = (1-np.exp(-self.__ADK(E[:, i].real, 2.05999, 3, 21.56*ev, 1.99547, 0.7943, 0).cumsum()*self.dt_atomic)) #Ne
        N_gas = self.__n_gas(p_g, P, 16.3*ev, self.w0_atomic)-self.__n_gas(p_g, 1, 8*ev, self.w0_atomic) #Ne
        N_kerr = self.__n_kerr(p_g, E[:, i], 1.31*10**(-24)) #Ne
      elif gas == 'Ar':
        P = 1-np.exp(-self.__ADK(E[:, i].real, 2.02870, 3, 15.85*ev, 1.24665, 0.92915, 0).cumsum()*self.dt_atomic) #Ar
        N_gas = self.__n_gas(p_g, P, 8*ev, self.w0_atomic) #Ar
        N_kerr = self.__n_kerr(p_g, E[:, i], 9.8*10**(-24)) #Ar
      elif gas == 'N2':
        P = 1-np.exp(-0.4*self.__ADK(E[:, i].real, 2.02870, 3, 15.85*ev, 1.24665, 0.92915, 0).cumsum()*self.dt_atomic) #N2
        N_gas = (0.0002953/0.0002789)*self.__n_gas(p_g, P, 8*ev, self.w0_atomic) #N2
        N_kerr = self.__n_kerr(p_g, E[:, i], 8*10**(-24)) #N2
      elif gas == 'Kr':
        P = 1-np.exp(-self.__ADK(E[:, i].real, 2.00636, 3, 14.35*ev, 1.04375, 0.98583, 0).cumsum()*self.dt_atomic) #Kr
        N_gas = self.__n_gas(p_g, P, 6.5*ev, self.w0_atomic) #Kr
        N_kerr = self.__n_kerr(p_g, E[:, i], 27.8*10**(-24)) #Kr
      W_plasma_square = self.__w_plasma_square(p_g, P)
      tau_i = int((np.pi/self.w0)/self.dt)
      W_plasma_square = np.array(pd.Series(W_plasma_square).rolling(window=tau_i, center=True).mean())
      W_plasma_square[-int(tau_i/2):] = np.nanmin(W_plasma_square)
      W_plasma_square[:int(tau_i/2)] = 0
      W_plasma_square = W_plasma_square/(1+np.exp((self.T-self.T.max()+2*np.pi/self.w0)*self.w0))
      if not neutral_disp:
        N_gas -= N_gas.max()
      self.Sw[:, i] = -2*np.fft.fft((N_gas+N_kerr)*E[:, i]/c**2)*self.Omega**2+np.fft.fft(W_plasma_square*E[:, i]/c**2)
      if r == 0:
        self.Pz.append(P)

  def __ADK(self, E, Cnl, Glm, Ip, F0, n, m):
    return Cnl**2*Glm*Ip*(2*F0/np.abs(E))**(2*n-np.abs(m)-1)*np.exp(-2*F0/(3*np.abs(E)))

  def __n_gas(self, p_g, p, Ie, w):
    return 2*np.pi*(2.5*10**(-5)*(1/angstrom)**3)*p_g*((1-p)/(Ie**2-w**2))

  def __w_plasma_square(self, p_g, p):
    return 4*np.pi*(2.5*10**(-5)*(1/angstrom)**3)*p_g*p*(10**(15)*fs)**2

  def __n_kerr(self, p_g, E, n2):
    return p_g*np.abs(E)**2*(3.51*10**(20))*n2

def unwrap_self_subcalc(arg, **kwarg):
  # メソッドfをクラスメソッドとして呼び出す関数
  return CEPScanner.subcalc(*arg, **kwarg)

class CEPScanner:

  def __init__(self, rmax=400*um, rnum=300, r0_beam=100*um, zrange=6000*um,
               f=4000*um, znum=400, tmax=25*fs_si, tnum=1000, l0=1.6*um,
               emax=280*mv_per_cm, e_fwhm=10*fs_si, wmax=2*np.pi*2*10**15, tnum_hhg=5000,
               nonlinear=True, gas='Ar', p_g=1, neutral_disp=True, hhg=True, CEPs=np.linspace(0, 1, 10)):
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
    self.wmax = wmax
    self.tnum_hhg = tnum_hhg
    self.nonlinear = nonlinear
    self.gas = gas
    self.p_g = p_g
    self.neutral_disp = neutral_disp
    self.hhg = hhg
    self.CEPs = CEPs
    self.deformator = WaveformDeformator(rmax=self.rmax, rnum=self.rnum, r0_beam=self.r0_beam, zrange=self.zrange,
                                      f=self.f, znum=self.znum, tmax=self.tmax, tnum=self.tnum, l0=self.l0,
                                      emax=self.emax, e_fwhm=self.e_fwhm, wmax=self.wmax, tnum_hhg=self.tnum_hhg, cep=0*np.pi)

  def check_convergence(self):
    self.deformator.propagate(nonlinear=self.nonlinear, gas=self.gas, p_g=self.p_g, neutral_disp=self.neutral_disp, wmax=self.wmax, hhg=False)

  def subcalc(self, cep):
    deformator = WaveformDeformator(rmax=self.rmax, rnum=self.rnum, r0_beam=self.r0_beam, zrange=self.zrange,
                                    f=self.f, znum=self.znum, tmax=self.tmax, tnum=self.tnum, l0=self.l0,
                                    emax=self.emax, e_fwhm=self.e_fwhm, wmax=self.wmax, tnum_hhg=self.tnum_hhg, cep=cep*np.pi)
    deformator.propagate(nonlinear=self.nonlinear, gas=self.gas, p_g=self.p_g, neutral_disp=self.neutral_disp, wmax=self.wmax, hhg=self.hhg)
    return deformator.X_hhg

  def propagate_CEP(self, max_processes=None):
    if max_processes is not None:
      processes = min(len(self.CEPs), max_processes)
    else:
      processes = len(self.CEPs)
    pool = mp.Pool(processes)
    self.X_hhgs = pool.map(unwrap_self_subcalc, zip([self]*len(self.CEPs), self.CEPs))

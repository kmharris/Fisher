import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse
from decimal import *

from astropy.cosmology.funcs1 import distmod
from statsmodels.stats import correlation_tools
#I'll need to put in cosmology because I took it all out of the source code

#small dmb


#MEANS
'''
M = -1.934137e+01 

omega_m  = 2.024139e-01

h0  = 70.00688

w =  -8.141489e-01

omega_b  =  0.04

b = 9.839909e-01  
sign1a = 6.166667e-02

'''

#best fit

M = -19.3698

omega_m  =  0.201148

h0  = 69.084

w =  -0.811708

omega_b  =  0.04

b = 0.981227
sign1a = 0.0558139


#lets do some mu calculations
from astropy.cosmology import Flatw0waCDM
#set a specific cosmology

cos = Flatw0waCDM(h0, omega_m, w0= w, Ob0 = omega_b)

datamat = np.loadtxt('small_dmb_test_out.txt', unpack=True)    


zhel = datamat[2] # redshift 
zcmb = datamat[1] # redshift
mb = datamat[4] 
dmb = datamat[5]
p1a = datamat[-1]





def f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a):
    cos = Flatw0waCDM(h0, omega_m, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value
    
    mu_obs= mb - float(M)
    
    diffvec1a = (mu_obs-mu_theory)
       
    diffvecn1a = (mu_obs - mu_theory - float(b))

    c1a = diffvec1a**2/dmb**2
    cnon1a = diffvecn1a**2/float(sign1a)**2
    
    rt = []
    for i in range(40):
        ps = Decimal(p1a[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*dmb[i]))*Decimal(-c1a[i]/2).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*float(sign1a)))*Decimal(-cnon1a[i]/2).exp()
        rt.append(float(ps*f1 + (Decimal(1.)-ps)*f2))
    
    return np.array(rt)

def log_f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a):
    cos = Flatw0waCDM(h0, omega_m, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value
    
    mu_obs= mb - float(M)
    
    diffvec1a = (mu_obs-mu_theory)
       
    diffvecn1a = (mu_obs - mu_theory - float(b))

    c1a = diffvec1a**2/dmb**2
    cnon1a = diffvecn1a**2/float(sign1a)**2
    #f1 = (1./(np.sqrt(2*np.pi)*dmb))*np.exp(-c1a/2)
    #f2 = (1./(np.sqrt(2*np.pi)*sign1a))*np.exp(-cnon1a/2)
    
    rt = []
    for i in range(40):
        ps = Decimal(p1a[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*dmb[i]))*Decimal(-c1a[i]/2).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*float(sign1a)))*Decimal(-cnon1a[i]/2).exp()
        rt.append(float(Decimal.ln(ps) + Decimal.ln(f1) + Decimal.ln(Decimal(1.) + (Decimal(1.)-ps)*f2/(ps*f1))))
    
    
    return np.array(rt)


    
#this outputs forty points, which is why the trapezoidal rule will work but the others won't
   
#just comment out what I don't need


#this is for M(x axis) and omega_m (y axis)
    
#Order of variables is M, h0, omega_m, w, b, sign1a

#first row 
    
def I_00(mb): #dm^2
    h = 1e-5 # #2.909426e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M + h, h0, omega_m,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M - h , h0, omega_m,   w, b, sign1a))/(2*h))**2

def I_01(mb): #dm*dh0
    hm = 1e-5 # 2.909426e-03
    hh = 1e-5 # 9.365970e-02
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,  w, b , sign1a ))/(2*hh))
 
def I_02(mb): #dm*domega_m
    hm = 1e-5 # 2.909426e-03
    hom = 1e-5 # 8.596853e-04
    
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom , w, b , sign1a ))/(2*hom))
 

def I_03(mb): #dm*dw
    hm = 1e-5 # 2.909426e-03
    hw = 1e-5 # 1.491923e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2*hw))
 
def I_04(mb): #dm*db
    hm = 1e-5 # 2.909426e-03
    hb = 1e-5 # 1.380365e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b + hb, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w , b -hb, sign1a ))/(2*hb))
 
def I_05(mb): #dm*dsign1a
    hm = 1e-5 # 2.909426e-03
    hs = 1e-5 # 5.466113e-06
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b , sign1a + hs) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w , b, sign1a - hs ))/(2*hs))
 
# second row

def I_10(mb): #dh0*dm
    return I_01(mb)

def I_11(mb): #dh0^2
    h = 1e-5 # 9.365970e-02
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0 + h, omega_m,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 - h, omega_m,   w, b, sign1a))/(2*h))**2

def I_12(mb): #dh0*domega_m
    hh = 1e-5 # 9.365970e-02
    hom = 1e-5 # 8.596853e-04
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m +hom,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m - hom,   w, b, sign1a))/(2*hom))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,  w, b , sign1a ))/(2*hh))
  
def I_13(mb): #dh0*dw
    hh = 1e-5 # 9.365970e-02
    hw = 1e-5 # 1.491923e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w + hw, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w - hw, b, sign1a))/(2*hw))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,   w, b , sign1a ))/(2*hh))

def I_14(mb): #dh0*db
    hh = 1e-5 # 9.365970e-02
    hb = 1e-5 # 2.661960e-04
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b +hb , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b - hb, sign1a))/(2*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,   w, b , sign1a ))/(2*hh))

def I_15(mb): #dh0*dsign1a
    hh = 1e-5 # 9.365970e-02
    hs = 1e-5 # 5.466113e-06
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b , sign1a + hs ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b, sign1a - hs))/(2*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,   w, b , sign1a ))/(2*hh))

#third row

def I_20(mb): #dM*domega_m
    return I_02(mb)

def I_21(mb): #dh0*domega_m
    return I_12(mb)

def I_22(mb): #domega_m^2
    h = 1e-5 # 8.596853e-04
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m + h,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m - h,   w, b, sign1a))/(2*h))**2

def I_23(mb): #domega_m*dw
    hom = 1e-5 # 8.596853e-04
    hw = 1e-5 # 1.491923e-03
    
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w + hw, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w - hw, b, sign1a))/(2*hw))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2*hom))
  
def I_24(mb): #domega_m*db
    hom = 1e-5 # 8.596853e-04
    hb = 1e-5 # 2.661960e-04
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b + hb , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b - hb, sign1a))/(2*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2*hom))
  
def I_25(mb): #domega_m*dsign1a
    hom = 1e-5 # 8.596853e-04
    hs = 1e-5 # 5.466113e-06
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b  , sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b , sign1a - hs))/(2*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2*hom))
  
#fourth row


def I_30(mb): #dM*dw
    return I_03(mb)

def I_31(mb): #dw*dh0
    return I_13(mb)

def I_32(mb): #dw*domega_m
    return I_23(mb)

def I_33(mb): #dw^2
    h = 1e-5 # 1.491923e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w + h, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w - h, b, sign1a))/(2*h))**2

def I_34(mb): #dw*db
    hw = 1e-5 # 1.491923e-03
    hb = 1e-5 # 2.661960e-04
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b +hb, sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b - hb, sign1a))/(2*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2*hw))

def I_35(mb): #dw*dsign1a
    hw =1e-5 #  1.491923e-03
    hs = 1e-5 # 5.466113e-06
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b, sign1a - hs))/(2*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2*hw))

#fifth row 
    
def I_40(mb): #dM*db
    return I_04(mb)

def I_41(mb): #dh0*db
    return I_14(mb)

def I_42(mb): #domega_m*db
    return I_24(mb)

def I_43(mb): #dw*db
    return I_34(mb)

def I_44(mb): #db^2
    h = 1e-5 # 2.661960e-04
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b + h, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w , b - h, sign1a))/(2*h))**2

def I_45(mb): #dsign1a*db
    hb = 1e-5 # 2.661960e-04
    hs = 1e-5 # 5.466113e-06
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w, b , sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M  , h0, omega_m ,   w, b, sign1a - hs))/(2*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b + hb, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m , w , b -hb, sign1a ))/(2*hb))
 
#sixth row 
 
def I_50(mb): #dsign1a*dM
    return I_05(mb)

def I_51(mb): #dsign1a*dh0
    return I_15(mb)

def I_52(mb): #dsign1a*domega_m
    return I_25(mb)

def I_53(mb): #dsign1a*dw
    return I_35(mb)

def I_54(mb): #dsign1a*db
    return I_45(mb)

def I_55(mb): #dsign1a^2
    #h = 1.011612e-03
    h = 1e-5 #  5.466113e-06
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b, sign1a +h ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w , b, sign1a - h ))/(2*h))**2



F = np.zeros((6,6)) #empty matrix that will become the fisher matrix

#x = np.arange(-50,50, 0.01) keep this as an idea for later -> I need a complete data set with more mb, zmb, and dmb

for i in range(6):
    for j in range(6):
        F[i,j] = np.trapz(eval('I_'+'{:1.0f}'.format(i) + '{:1.0f}'.format(j))(mb), mb) #uses trapz to integrate over data vector mb


C = np.linalg.inv(F)
#C = correlation_tools.cov_nearest(C)

C_M_omega_m = np.zeros((2,2))
C_M_omega_m[0,0] = C[0,0]
C_M_omega_m[0,1] = C[0,2]
C_M_omega_m[1,0] = C_M_omega_m[0,1]
C_M_omega_m[1,1] = C[2,2]

C_M_w = np.zeros((2,2))
C_M_w[0,0] = C[0,0]
C_M_w[0,1] = C[0,3]
C_M_w[1,0] = C_M_w[0,1]
C_M_w[1,1] = C[3,3]


C_w_omega_m = np.zeros((2,2))
C_w_omega_m[0,0] = C[3,3]
C_w_omega_m[0,1] = C[2,3]
C_w_omega_m[1,0] = C_w_omega_m[0,1]
C_w_omega_m[1,1] = C[2,2]

C_sign1a_b = np.zeros((2,2))
C_sign1a_b[0,0] = C[5,5]
C_sign1a_b[0,1] = C[4,5]
C_sign1a_b[1,0] = C_sign1a_b[0,1]
C_sign1a_b[1,1] = C[4,4]

C_sign1a_omega_m = np.zeros((2,2))

C_sign1a_omega_m[0,0] = C[5,5]
C_sign1a_omega_m[0,1] = C[2,5]
C_sign1a_omega_m[1,0] = C_sign1a_b[0,1]
C_sign1a_omega_m[1,1] = C[2,2]

C_b_w = np.zeros((2,2))

C_b_w[0,0] = C[4,4]
C_b_w[0,1] = C[4,3]
C_b_w[1,0] = C_b_w[0,1]
C_b_w[1,1] = C[3,3]

C_b_omega_m = np.zeros((2,2))

C_b_omega_m[0,0] = C[4,4]
C_b_omega_m[0,1] = C[4,2]
C_b_omega_m[1,0] = C_b_omega_m[0,1]
C_b_omega_m[1,1] = C[2,2]

# change this, centre, range, and labels as needed

w,v=np.linalg.eigh(C_w_omega_m)

angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
angle = -118

#

a_1s=np.sqrt(2.3*np.abs(w[0])) #68% #it seems like a is whatever the 2nd element of the array is, so sigy**2
b_1s=np.sqrt(2.3*w[1])
a_2s=np.sqrt(6.17*np.abs(w[0])) #95%
b_2s=np.sqrt(6.17*w[1])
a_3s = np.sqrt(11.8*np.abs(w[0]))
b_3s = np.sqrt(11.8*w[1])


#centre=np.array([params[i1].val,params[i2].val])


#make cuts 
w = -0.811708
centre = np.array([w , omega_m])

e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '68% confidence')
#                     

e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '95% confidence')
#                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])

e_3s=Ellipse(xy=centre,width=2*a_3s,height=2*b_3s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='purple',  label = '99% confidence')


ax = plt.gca()

M_omega_m_emcee = np.loadtxt('M_omega_m_emcee_simplebeams.txt', unpack=True)  

full_emcee = np.loadtxt('full_emcee_simplebeams.txt', unpack=True)  

small_dmb = np.loadtxt('small_dmb_emcee_simplebeams.txt', unpack=True)

#plt.axis([0.97, 1.0, 0.2, 0.3] ) #b, omega_m
#plt.axis([0.0, 0.5, 0.8, 1.2] ) #sign1a, b
#plt.axis([0, 0.1, 0.1, 0.4] ) #sign1a, omega_m
#plt.axis([-19.39, -19.29, -1.4, -0.6] ) #M, w
plt.axis([ -.9, -0.725, 0.16, 0.24] ) #w, omega_m
#plt.axis([0.8,1.2,-.9,-.725]) #b, w



plt.plot()

plt.xlabel('$w$')
plt.ylabel("$\Omega_{m}$")

ax.add_patch(e_1s)
ax.add_patch(e_2s)
#ax.add_patch(e_3s)

# order is omega_m, h0, w, M, b, sign1a 
ax.scatter(small_dmb[2][8000:-1],small_dmb[0][8000:-1], c = 'purple', s = 0.2, alpha = 0.1)
ax.legend()
plt.savefig('gaussian_bam.pdf')
plt.show()


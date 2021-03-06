import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse
import random as random
import os

data_file = os.path.join(os.path.expanduser('~/cosmosis/ini_files'), 'gaussian_data.txt')

data = np.loadtxt(data_file, unpack = True)


'''
This code uses two gaussian distributions generated by data editor
to test the functionality of the mixed gaussian
fisher matrix code
'''

#fill in sampled over values with their best fit values

x =  data[0] #sorts data from smallest to largest 
#sig1 = 0.01
sig2 = 0.0472074
#u1 = 0.
u2 =0.979568
alpha = 1.99998
y = data[1]
dy = data[2]
p1 = data[3]    

x_og = []
x_shift = []

for i in range(0,len(x)):
    if p1[i]>0.2:
        x_og.append(int((np.where(x == x[i]))[0]))
    else:
        x_shift.append(int((np.where(x == x[i]))[0]))


'''
#original
def f(xs,u1s,u2s,p1s,sig1s,sig2s): #function that represents the mixed gaussian, acts as BEAMS posterior

    rt = []
    for i in range(len(x)):
        ps = Decimal(p1s[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi))*(sig1s))*Decimal(-(xs[i]-u1s)**2/(2*sig1s**2)).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi))*(sig2s))*Decimal(-(xs[i]-u2s)**2/(2*sig2s**2)).exp()

        rt.append(float(ps*f1 + (Decimal(1.)-ps)*f2))
    
    return np.array(rt)

def log_f(xs,u1s,u2s,p1s,sig1s,sig2s): #log of that function
    rt = []
    for i in range(len(x)):
        ps = Decimal(p1[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*(sig1s)))*Decimal(-(xs[i]-u1s)**2/(2*sig1s**2)).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*(sig2s)))*Decimal(-(xs[i]-u2s)**2/(2*sig2s**2)).exp()
        #print 'f1', f1
        #print 'f2', f2
        #print ''
        
        rt.append(float(Decimal.ln(ps) + Decimal.ln(f1) + Decimal.ln(Decimal(1.) + (Decimal(1.)-ps)*f2/(ps*f1))))
    
    return np.array(rt)
    
'''

def f(xs,p1s,ys,dys,u2s,sig2s,alphas): #function that represents the mixed gaussian, acts as BEAMS posterior

    rt = []
    for i in range(len(xs)):
        ps = Decimal(p1s[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi))*(dy[i]))*Decimal(-(ys[i] - alphas*xs[i])**2/(2*dy[i]**2)).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi))*(dy[i]))*Decimal(-(ys[i] - alphas*xs[i] - u2s)**2/(2*sig2s**2)).exp()

        rt.append(float(ps*f1 + (Decimal(1.)-ps)*f2))
    
    return np.array(rt)

def log_f(xs,p1s,ys,dys,u2s,sig2s, alphas): #log of that function
    rt = []
    for i in range(len(xs)):
        ps = Decimal(p1s[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi))*(dy[i]))*Decimal(-(ys[i] - alphas*xs[i])**2/(2*dy[i]**2)).exp()
        print f1
        
        f2 = Decimal(1./(np.sqrt(2*np.pi))*(dy[i]))*Decimal(-(ys[i] - alphas*xs[i] - u2s)**2/(2*sig2s**2)).exp()

        print f2
        print ''
        rt.append(float(Decimal.ln(ps) + Decimal.ln(f1) + Decimal.ln(Decimal(1.) + (Decimal(1.)-ps)*f2/(ps*f1))))
    
    return np.array(rt)

#this naming convention is important, do dlogfd + whatever the name of the parameter is 
'''
def dlogfdu1(x):
    h = 1e-8#u2
 #arbitrarily small step for analytical derivative
    return (log_f(x,u1 +h,u2,p1,sig1,sig2) - log_f(x,u1,u2,p1,sig1,sig2))/h

def dlogfdu2(x): 
    h = 1e-8
    #arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2 + h,p1,sig1,sig2) - log_f(x,u1,u2 - h,p1,sig1,sig2))/(2*h)

def dlogfdp1(x):
    h = 1e-8#u2
 #arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2 ,p1 +h,sig1,sig2) - log_f(x,u1,u2,p1,sig1,sig2))/h

def dlogfdsig1(x): #u2
    h = 1e-4#arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2 ,p1,sig1 + h,sig2) - log_f(x,u1,u2,p1,sig1,sig2))/h

def dlogfdsig2(x): #u2
    h = 1e-8#arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2,p1,sig1,sig2 +h ) - log_f(x,u1,u2,p1,sig1,sig2-h))/(2*h)    
'''

def dlogfdu2(x):
    #h = 1e-8
    return (log_f(x,p1,y,dy,u2 + h, sig2, alpha) - log_f(x,p1,y,dy,u2,sig2,alpha))/h

def dlogfdsig2(x):
    #h = 1e-8
    return (log_f(x,p1,y,dy,u2, sig2 +h , alpha) - log_f(x,p1,y,dy,u2,sig2,alpha))/h

def dlogfdalpha(x):
    #h = 1e-8
    return (log_f(x,p1,y,dy,u2,sig2, alpha + h) - log_f(x,p1,y,dy,u2,sig2,alpha))/h



u2s = []
sig2s = []
alphas = []

hs = np.logspace(-9, 0, 100, base = 10.)
for h in hs:
    u2s.append(np.trapz((eval('dlogfdu2')(x))**2*f(x,p1,y,dy,u2,sig2,alpha), x))
    alphas.append(np.trapz((eval('dlogfdalpha')(x))**2*f(x,p1,y,dy,u2,sig2,alpha), x))


plt.plot(hs, u2s, 'ro')
plt.xscale('log')
plt.yscale('log')
plt.show()

'''
funcs = ['dlogfdu2','dlogfdalpha']

I = np.zeros((len(funcs),len(funcs)))


for i in range(len(funcs)):
    for j in range(len(funcs)):
        I[i,j] = np.trapz(eval(funcs[i])(x)*eval(funcs[j])(x)*f(x,p1,y,dy,u2,alpha), x)

'''
'''
ind = [1,3]
non_ind = [0,2]

def margMat(inds,non_inds, FMin):
 
    #We will use the F = A - B C B^-1 formalism from the appendix of 
    #Matsubara (arXiv:astro-ph/0408349) to obtain the marginalised matrix 


    size = np.shape(FMin)[0]
    A = np.zeros((len(inds),len(inds))) # the matrix of our parameters
    B = np.zeros((len(inds),size-len(inds)))
    C = np.zeros((size-len(inds), size-len(inds))) # the matrix of nuisance parameters
    
    
# Computing A, B, C
    print inds

    for i in range(len(inds)):
        for j in range(len(inds)):
            A[i,j] = FMin[int(inds[i]),int(inds[j])]
            
    for i in range(len(non_inds)):
        for j in range(len(non_inds)):
            C[i,j] = FMin[int(non_inds[i]), int(non_inds[j])]
                    
    for i in range(len(inds)):
        for j in range(len(non_inds)):
            B[i,j] = FMin[int(inds[i]),int(non_inds[j])]

    A = np.mat(A)
    B = np.mat(B)
    C = np.mat(C)


    FMout = A - B*C.I*B.T

    return FMout

FM_out = margMat(ind, non_ind, I)
'''

#F = np.linalg.inv(I)

    

def plotting(a,b): #where a and b correspond to the parameter you want in quotes ie. "u2"
    a = [i for i,s in enumerate(funcs) if a in s][0]
    b = [i for i,s in enumerate(funcs) if b in s][0]
    
    B = np.zeros((2,2))
    B[0,0] = F[a,a]
    B[1,0] = F[a,b]
    B[0,1] = F[b,a]
    B[1,1] = F[b,b]
    
    w,v=np.linalg.eigh(B)
    angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
    
    print w
    
    a_1s=np.sqrt(2.3*np.abs(w[0])) #68% confidence
    b_1s=np.sqrt(2.3*np.abs(w[1]))
    a_2s=np.sqrt(6.17*np.abs(w[0])) #95% confidence 
    b_2s=np.sqrt(6.17*np.abs(w[1]))
      
    centre = np.array([eval(funcs[int(a)].split('dlogfd')[1]),eval(funcs[int(b)].split('dlogfd')[1])])
    
    print a_1s
    print b_1s
    
    e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '$68$% $confidence$')
    #                     
    
    e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '$95$% $confidence$')
    #                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])
    
   
    



    if -0.02 < np.cos(np.deg2rad(angle))  < 0.02:
        plt.axis([centre[0]- 2*b_2s, centre[0] + 2*b_2s, centre[1] - 2*a_2s, centre[1]+ 2*a_2s]) 
        plt.plot()
    elif -0.02 < np.sin(np.deg2rad(angle))  < 0.02 :
        plt.axis([centre[0]- 2*a_2s, centre[0] + 2*a_2s, centre[1] - 2*b_2s, centre[1]+ 2*b_2s]) 
        plt.plot()
        
    else:
        plt.axis([centre[0]- 2*np.cos(np.deg2rad(angle))*a_2s, centre[0] + 2*np.cos(np.deg2rad(angle))*a_2s, centre[1] - 2*np.cos(np.deg2rad(angle))*b_2s, centre[1]+ 2*np.cos(np.deg2rad(angle))*b_2s]) 
        plt.plot()
        


     
    #emcee order is b, sign1a -> this part will be uncommented as you need it
    #you can use it to compare sign1a and b
   
    emcee_file = os.path.join(os.path.expanduser('~/cosmosis/'), 'gaussian.txt')
    
    emcee = np.loadtxt(emcee_file, unpack = True)
    
    ax = plt.gca()
    
    #plt.axis([min(emcee[a]), max(emcee[a]), min(emcee[b]), max(emcee[b])]) 
    #plt.plot()
    
    
   
    plt.xlabel(funcs[int(a)].split('dlogfd')[1])
    plt.ylabel(funcs[int(b)].split('dlogfd')[1])
    ax.scatter(emcee[a][24000:-1], emcee[b][24000:-1],  c = 'purple', s = 0.2, alpha = 0.1) #this plots the MCMC point from CosmoSIS if you have them to compare to
    ax.add_patch(e_1s)
    ax.add_patch(e_2s)
    
    
    ax.legend()
    
    plt.show()




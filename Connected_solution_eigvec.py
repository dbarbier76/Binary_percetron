import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import time
import math 


import warnings
warnings.filterwarnings("ignore")


pi=np.arccos(-1)
eps_=10**-20

e_  = lambda x: np.exp(x)
log_= lambda x: np.log(x)

sinh_=lambda x: np.sinh(x)
cosh_=lambda x: np.cosh(x)
tanh_=lambda x: np.tanh(x)
sqt_= lambda x: np.sqrt(abs(x)+eps_)

erf_= lambda x: special.erf(x)
derf_=lambda x:  (2/sqt_(pi))*e_(-x*x)
derf_2=lambda x: (2/sqt_(pi))*(-2*x)*e_(-x*x)

Var1=lambda x,y,z: (x-y)/sqt_(2*z)
dx_Var1=lambda x,y,z: 1/sqt_(2*z)
dy_Var1=lambda x,y,z: -1/sqt_(2*z)
dz_Var1=lambda x,y,z: (-1/2)*(x-y)/(sqt_(2*z)*z)

Var2=lambda x,y,z: (x+y)/sqt_(2*z)
dx_Var2=lambda x,y,z: 1/sqt_(2*z)
dy_Var2=lambda x,y,z: 1/sqt_(2*z)
dz_Var2=lambda x,y,z: (-1/2)*(x+y)/(sqt_(2*z)*z)

H=    lambda x,y,z:(1/2)*erf_(Var1(x,y,z))+(1/2)*erf_(Var2(x,y,z)) #if np.sign(abs(Var1(x,y,z))-5.5)+np.sign(abs(Var2(x,y,z))-5.5)<1.5  else  (      np.sign(Var1(x,y,z))/2+np.sign(Var2(x,y,z))/2 -e_(-Var1(x,y,z)**2)/(2*sqt_(pi)*(Var1(x,y,z)))-e_(-Var2(x,y,z)**2)/(2*sqt_(pi)*(Var2(x,y,z)))     ) 
dx_H_=lambda x,y,z:(1/(2*sqt_(2*z)))*(+                         derf_((x-y)/sqt_(2*z))  +                         derf_((x+y)/sqt_(2*z)) )
dy_H_=lambda x,y,z:(1/(2*sqt_(2*z)))*(-                         derf_((x-y)/sqt_(2*z))  +                         derf_((x+y)/sqt_(2*z)) )
dz_H_=lambda x,y,z:(1/2)            *(-((x-y)/((2*z)**(3/2))) * derf_((x-y)/sqt_(2*z))  -((x+y)/((2*z)**(3/2))) * derf_((x+y)/sqt_(2*z)) ) 


log_H=lambda x,y,z   :   log_(H(x,y,z)) #if np.sign(abs(Var1(x,y,z))-5.5)+np.sign(abs(Var2(x,y,z))-5.5)<1.5         else max(   -log_(2*sqt_(pi)*abs(Var1(x,y,z)))-Var1(x,y,z)**2                                  ,  -log_(2*sqt_(pi)*abs(Var2(x,y,z)))-Var2(x,y,z)**2   )
dx_log_H=lambda x,y,z:   dx_H_(x,y,z)/H(x,y,z) #if np.sign(abs(Var1(x,y,z))-5.5)+np.sign(abs(Var2(x,y,z))-5.5)<1.5  else max(-np.sign(Var1(x,y,z))*dx_Var1(x,y,z)/abs(Var1(x,y,z))-2*dx_Var1(x,y,z)*Var1(x,y,z)    ,  -np.sign(Var2(x,y,z))*dx_Var2(x,y,z)/abs(Var2(x,y,z))-2*dx_Var2(x,y,z)*Var2(x,y,z))
dy_log_H=lambda x,y,z:   dy_H_(x,y,z)/H(x,y,z) #if np.sign(abs(Var1(x,y,z))-5.5)+np.sign(abs(Var2(x,y,z))-5.5)<1.5  else max(-np.sign(Var1(x,y,z))*dy_Var1(x,y,z)/abs(Var1(x,y,z))-2*dy_Var1(x,y,z)*Var1(x,y,z)    ,  -np.sign(Var2(x,y,z))*dy_Var2(x,y,z)/abs(Var2(x,y,z))-2*dy_Var2(x,y,z)*Var2(x,y,z))
dz_log_H=lambda x,y,z:   dz_H_(x,y,z)/H(x,y,z) #if np.sign(abs(Var1(x,y,z))-5.5)+np.sign(abs(Var2(x,y,z))-5.5)<1.5  else min(-np.sign(Var1(x,y,z))*dz_Var1(x,y,z)/abs(Var1(x,y,z))-2*dz_Var1(x,y,z)*Var1(x,y,z)    ,  -np.sign(Var2(x,y,z))*dz_Var2(x,y,z)/abs(Var2(x,y,z))-2*dz_Var2(x,y,z)*Var2(x,y,z))    
    

f=       lambda κ,ξ,x: (κ/tanh_(ξ))*tanh_(ξ*x)
deriv_f= lambda κ,ξ,x: (κ/tanh_(ξ))*ξ*(1-tanh_(ξ*x)*tanh_(ξ*x))


def potential(κ_new,κ_old,α,m,P_x,P_y):
    
    P = lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(w,P_x,P_y) 
    N_w=integrate.quad(P, -κ_old,κ_old)[0]
    
    entropic_term=-log_((1-m)/2)*(1-m)/2-log_((1+m)/2)*(1+m)/2
    
    var=  lambda w:(1/2)*erf_((κ_new-m*w)/sqt_(2*(1-m*m)))+(1/2)*erf_((κ_new+m*w)/sqt_(2*(1-m*m)))
    func= lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(w,P_x,P_y) *log_(var(w))

    cut=max(0,κ_old-50*sqt_(2*(1-m*m)))
    
    energetic_term=(α/N_w)*integrate.quad(func, cut ,κ_old )[0]+(α/N_w)*integrate.quad(func,-κ_old,-cut )[0]+(α/N_w)*integrate.quad(func,-cut,cut )[0]

        
    return entropic_term+energetic_term



def Energ_term(κ_new,κ_old,α,m,P_x,P_y):
    
    var=  lambda w:(1/2)*erf_((κ_new-m*w)/sqt_(2*(1-m*m)))+(1/2)*erf_((κ_new+m*w)/sqt_(2*(1-m*m)))
    func= lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(w,P_x,P_y) *log_(var(w))

    cut=max(0,κ_old-30*sqt_(2*(1-m*m)))
    
    energetic_term=integrate.quad(func, cut ,κ_old )[0]+integrate.quad(func,-κ_old,-cut )[0]+integrate.quad(func,-cut,cut )[0]

    return energetic_term

def potential_t(α,m,t,decomposition_list,Norm,decay_list,pot_list):
    
    entropic_term=-log_((1-m)/2)*(1-m)/2-log_((1+m)/2)*(1+m)/2
    energetic_term=(α/Norm)*np.sum(decomposition_list[i]*pot_list[i]*(decay_list[i]**t) for i in range(len(decomposition_list)))
    
    return entropic_term+energetic_term






def potential_dynamics(P_y,α,m,N_time,decomposition_list,norms_list,decay_list,pot_list):
    pot_dynamics=np.zeros(N_time)
    t_max=0
    
    for t in range(N_time):
        t_real=t*0.01
        
        Norm=sum(  decomposition_list[i]*norms_list[i]*(decay_list[i]**t_real)       for i in range(len(decomposition_list)))
        pot_dynamics[t]=potential_t(α,m,t_real,decomposition_list,Norm,decay_list,pot_list)
        
        if pot_dynamics[t]<-0.001*abs(pot_list[np.argmax(decay_list)]):
            break
        t_max=t_real
        
    if t_max!=0:
        pow_=1.2
        cost=(abs(P_y)**pow_)*(1-np.sign(P_y))
        return pot_dynamics,m**t_max,m**t_max+np.sum(cost)
    else :
        return pot_dynamics,m**t_max,m**t_max+abs(pot_dynamics[0])*40

def optimization_planting_routine(P_x,P_y,α,m,N_time,decomposition_list_guess,norms_list,decay_list,pot_list):
    
    def deccorelation(decomposition_list):
        P_test=np.sum(P_y[i][:]*decomposition_list[i] for i in range(len(decomposition_list)))  
        pot,mo,cost=potential_dynamics(P_test,α,m,N_time,decomposition_list,norms_list,decay_list,pot_list)
        
        
        P = lambda w: np.interp(w,P_x,P_test) 
        N_plot=250
        x_list=np.zeros(N_plot)
        y_list=np.zeros(N_plot)
        for k in range(N_plot):
            x_list[k]=P_x[0]+(-P_x[0]+P_x[len(P_x)-1])*(k/N_plot)
            y_list[k]=P(x_list[k])
        plt.plot(x_list,y_list,label=str(round(cost,4))+'  '+str(round(mo,4)))
        plt.title(str(len(decomposition_list)))
        plt.legend()
        plt.show()
        
        return cost
    
    
    return optimize.fmin(deccorelation,decomposition_list_guess)
       
def routine_optimization(κ,α,m,decomposition_list_guess,N_vec,w,v,P_x):    
    
    ind = np.argpartition(w, -N_vec)[-N_vec:]
    P_y=np.zeros((N_vec,No_tot))

    ɸ=np.zeros(N_vec)
    decay_rate=np.zeros(N_vec)
    norms=np.zeros(N_vec)

    for i in range(N_vec):
        P_y[i][:]=v[:,ind[i]]

        ɸ[i]=Energ_term(κ,κ,α,m,P_x,P_y[i][:])
        decay_rate[i]=w[ind[i]]

        P = lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(w,P_x[:],P_y[i][:]) 
        norms[i]=integrate.quad(P, -κ,κ)[0]
        
    
    decomposition_list_opt=optimization_planting_routine(P_x,P_y,α,m,10**5,decomposition_list_guess,norms,decay_rate,ɸ)
    
    return decomposition_list_opt
    
    
    



#### Parameter of the problem ####
α=0.05

κ_min=0
κ_max=15
N_κ_try=17


N_size=0         #Number of system's size you wanna try
log_size_min=2    #Miniumum system's size in log_10 scaling
log_size_max=17   #Maximum  system's size in log_10 scaling


κ_crit=np.zeros(N_size)
N_list=np.zeros(N_size)



#### Parameter for the discretization of the distribution of interactions ####
No=250
No_tot=2*No
discretization="rescaled"
approx='No'



for k in range(N_size):

    # Set the system's size
    N=int(10**(log_size_min+(log_size_max-log_size_min)*(k/N_size)))
    N_list[k]=N#(log_size_min+(log_size_max-log_size_min)*(k/N_size))
    m=1-2/N



    ##################################################################################################
    ######### We compute the scaling function to get a good discretization close to the edge #########
    ##################################################################################################
    def slope(κ):
        def func(ξ):
            eq=deriv_f(κ,ξ,1)-5*sqt_(1-m*m)
            
            return eq
        
        ξo=optimize.fsolve(func,0.1)[0]
        return ξo
    
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    
    
    
    pot=0
    κ_minus=κ_min
    κ_plus=κ_max
    
    for l in range(N_κ_try):
        if l==0:
            κ=(κ_minus+κ_plus)/2

            if N>10**9:
                approx='Yes'
        else:
            
            if pot<0:
                κ_minus=κ
                κ=(κ+κ_plus)/2
            if pot>0:
                κ_plus=κ
                κ=(κ+κ_minus)/2
                
        
        ξo=slope(κ)



        # A is the update matrix to go from P^{t-1}[w] to P^{t}[w] where we added the symmetry w→-w (u_j=→-u_j) to lift degeneracy problems
        A=np.zeros((No_tot,No_tot))
    
        u_plus=np.zeros(No)
        u_minu=np.zeros(No)
        Δ_list=np.zeros(2*No)
    
        Po=np.ones(No_tot)
    
        if discretization=="rescaled":
            for i in range(No):
                u_plus[i]=+f(κ,ξo,(i+1)/No)
                u_minu[i]=-f(κ,ξo,(No-1-i)/No)
    
            u_tot=np.append(u_minu,u_plus)
        
        
        if discretization=="normal":    
            for i in range(No_tot):
                u_tot[i]=-κ+2*κ*((i)/(No_tot-1))

    
        for i in range(No_tot):
            if i==0:
                Δ_list[i]=abs(-κ-u_tot[0])
            else:
                Δ_list[i]=abs(u_tot[i-1]-u_tot[i])

   
        print(k,l)
        print("step 1: We compute the update matrix A")
    
        for i in range(No_tot):
            u_i=u_tot[i]
            
            center=-m*u_i
            center_index=np.argmin(abs(u_tot-center))
            center_index_2=center_index+1

            func= lambda v: (  e_(-(v+m*u_i)**2/(2*(1-m*m))) /sqt_(2*pi*(1-m*m))  )/H(κ,m*v,1-m*m) 
            
            for j in range(No_tot):
            
                u_j=u_tot[j]
                Δ=Δ_list[j]
                
            
                if abs(u_j-center)>15*sqt_(1-m*m) and j!=center_index and j!=center_index_2:
                    if approx=='No':
                        A[i][j]=integrate.quad(func, u_j-Δ,u_j )[0]
                    if approx=='Yes':
                        A[i][j]=0
                        

                if abs(u_j-center)<15*sqt_(1-m*m) and j!=center_index and j!=center_index_2:
                    A[i][j]=integrate.quad(func, u_j-Δ,u_j )[0]
                
                
                
                
                
                if j==center_index or j==center_index_2:
                    
                    if abs(u_j-u_tot[min(j+1,2*No-1)])>15*sqt_(1-m*m) and abs(u_j-u_tot[max(j-1,0)])>15*sqt_(1-m*m):
                        if approx=='No':
                            A[i][j]=integrate.quad(func, u_j-Δ,u_j )[0]
                        if approx=='Yes':
                            A[i][j]=1/(2*H(κ,m*u_j,1-m*m))
                    else:
                        A[i][j]=integrate.quad(func, u_j-Δ,u_j )[0]

        



            
        print("step 2: We get the eigenvectors of A")
        w,v=    np.linalg.eig(A) 
        
        print("step 3: We test if we are in the trivialization regime")
        trivial=0
        index_max=np.argmax(w) 
        

        P_x=u_tot        
        pot=potential(κ,κ,α,m,P_x,abs(v[:,index_max]))
        
        if l==N_κ_try-1:
            
            if pot<0:
                κ_minus=κ
                κ=(κ+κ_plus)/2
            if pot>0:
                κ_plus=κ
                κ=(κ+κ_minus)/2
                
            κ_crit[k]=κ
            
        print('log(N):',log_(N)/log_(10))
        print("kappa(plus/minus):",κ_plus,κ_minus)
        print("the cut is:",15*sqt_(1-m*m))
        print("kappa,pot:",κ,pot)            
        print("")  
          
    
    if pot>0:
        print("We are in the trivial regime just take the top eigenvector as your distribution !!!")
        trivial=1
    






print(N_list)
print(κ_crit)




plt.legend()
plt.show()

### Results for ⍺=0.05
α=0.05
log_size_min=2    #Miniumum system's size in log_10 scaling
log_size_max=16.5   #Maximum  system's size in log_10 scaling
N_list=[1.e+02,     1.e+03,     1.e+04,      1.e+05,    1.e+06,     1.e+07,     1.e+08,     1.e+09,     1.e+10,     1.e+11,     1.e+12,     1.e+13,     1.e+14,     1.e+15,     1.e+16]
κ_crit=[0.08623123, 0.15684128, 0.32186508, 0.70901871, 1.26932144, 1.81715012, 2.29013443, 2.69891739, 3.05929184, 3.38682175, 3.68722916, 3.966465, 4.2285347,  4.47526932, 4.6928215 ]

### Results for ⍺=0.5
α=0.5
log_size_min=2    #Miniumum system's size in log_10 scaling
log_size_max=16.5   #Maximum  system's size in log_10 scaling
N_list=[1.e+02,     1.e+03,     1.e+04,      1.e+05,    1.e+06,     1.e+07,     1.e+08,     1.e+09,     1.e+10,     1.e+11,     1.e+12,     1.e+13,     1.e+14,     1.e+15,     1.e+16]
κ_crit=[0.60370941, 0.98045807, 1.48751373, 1.97527924, 2.40674896, 2.78861542, 3.12973632, 3.44684295, 3.73658905, 4.0101944,  4.26726532, 4.51095123, 4.74322052, 4.96446686, 5.1585495 ]



func_scaling=lambda log_N,a,b: sqt_(max(0,a+b*log_N))
func_trivial=lambda log_N: sqt_(2*log_(2*α)+2*log_N)
def solve_scaling(x_ab):
    a=x_ab[0]
    b=x_ab[1]
    
    eq1=κ_crit[9] -func_scaling(log_(N_list[9]) ,a,b)
    eq2=κ_crit[14]-func_scaling(log_(N_list[14]),a,b)
    
    return [eq1,eq2]
out=optimize.fsolve(solve_scaling,[0.1,0.1])
print(out)



N_size=200
N_list_new=np.zeros(N_size)
κ_crit_interpol=np.zeros(N_size)
κ_crit_trivial=np.zeros(N_size)
for k in range(N_size):
    N_list_new[k]=int(10**(log_size_min+(log_size_max-log_size_min)*(k/N_size)))
    κ_crit_interpol[k]=func_scaling(log_(N_list_new[k]),out[0],out[1])
    κ_crit_trivial[k]=func_trivial(log_(N_list_new[k]))
    
    

plt.plot(log_(N_list[:])/log_(10),κ_crit[:],label='kappa crit.',linewidth=3.)
plt.plot(log_(N_list_new[:])/log_(10),κ_crit_interpol,c='black',linestyle='--')
plt.plot(log_(N_list_new[:])/log_(10),κ_crit_trivial,c='grey',linestyle='dotted')
plt.xlabel(r'$\log_{10}(N)$')
plt.ylabel(r'$\kappa_{\rm cluster}$')
plt.legend()
plt.show()


plt.plot(sqt_(log_(N_list[:])),κ_crit[:],label='kappa crit.',linewidth=3.)
plt.plot([log_(100)/log_(10),log_(1000000)/log_(10)],[1.9,1.9])
plt.plot([log_(5000)/log_(10),log_(5001)/log_(10)],[1.9,1.9],linewidth=5.)
plt.plot([log_(20000)/log_(10),log_(20010)/log_(10)],[1.9,1.9],linewidth=5.) 
plt.plot(log_(N_list_new[:])/log_(10),κ_crit_trivial,c='grey',linestyle='dotted')
plt.xlabel(r'$\log_{10}(N)$')
plt.ylabel(r'$\kappa_{\rm cluster}$')
plt.xlim([2,6])
plt.ylim([0,3])
plt.legend()
plt.show()
        






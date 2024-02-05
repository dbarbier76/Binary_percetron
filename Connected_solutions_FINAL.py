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

 
    
###### Computing κ_SAT ######
#############################

def κ_SAT_finding(α):
    def α_critical(κ_):
        κ=abs(κ_)
        integral=integrate.quad(lambda z: (np.exp(-z*z/2)/np.sqrt(2*pi)), -κ, κ)[0]
        out= α+np.log(2)/np.log(integral)
        
        return  abs(out)
    
    κo=(np.sqrt(2*pi)/2)*np.exp(-np.log(2)/α)
    κ_SAT=abs(optimize.fmin(α_critical,κo))
    κ_SAT=abs(optimize.fsolve(α_critical,κ_SAT))
    return κ_SAT

#############################
#############################


##################################
###### Originial transition ######
##################################

def Running_original_state_evolution(α,κ_old,κ_max,κ_min,Nκ,write):
        
    if write==1:
        file=open('Original_state_evolution(κ_old='+str(κ_old)+', α='+str(α)+').txt','w')
        file.write("κ_old	κ_new	q	m	q_hat	m_hat")
        file.write("\n")
        
        
    dκ=(κ_max-κ_min)/Nκ
    Nt=2000
    breaking=0
    
    for k in range(Nκ):
            κ_new=κ_min+k*dκ
            
            if k==0:
                q,m,q_hat,m_hat,cut=State_evolution_original([Nt,0.99999372, 0.999993553,κ_old,κ_new,α]) 
            else:
                if breaking==0:
                    q,m,q_hat,m_hat,cut=State_evolution_original([Nt,q,m,κ_old,κ_new,α]) 

                    
            if q<0.8:
                breaking=1
                      
            if write==1:
                    file.write(str(κ_old))
                    file.write("	")
                    file.write(str(κ_new))
                    file.write("	")
                    file.write(str(q))
                    file.write("	")
                    file.write(str(m))
                    file.write("	")
                    file.write(str(q_hat))
                    file.write("	")
                    file.write(str(m_hat))
                    file.write("\n")
                
    if write==1:            
        file.close()                

def test_ex(q,m,κ_old,κ_new,α,cut):
    q_=q+(10**-3)*(1-q)
    dq=(10**-3)*(1-q)
    
    var1= lambda B: (κ_old+B)/np.sqrt(2*(q-m*m))
    var2= lambda B: (κ_old-B)/np.sqrt(2*(q-m*m))
    var1_= lambda B: (κ_old+B)/np.sqrt(2*(q_-m*m))
    var2_= lambda B: (κ_old-B)/np.sqrt(2*(q_-m*m))
    
    var11= lambda B: (κ_new+B)/np.sqrt(2*(1-q))
    var22= lambda B: (κ_new-B)/np.sqrt(2*(1-q))
    var11_= lambda B: (κ_new+B)/np.sqrt(2*(1-q_))
    var22_= lambda B: (κ_new-B)/np.sqrt(2*(1-q_))
    
    func1= lambda B: (np.exp(-B*B/2)/np.sqrt(2*pi))   *   (1/2)*( special.erf(var1(B))  +  special.erf(var2(B)) )   *\
                                                   np.log((1/2)*( special.erf(var11(B)) +  special.erf(var22(B)) ))
    

    func2= lambda B: (np.exp(-B*B/2)/np.sqrt(2*pi))   *   (1/2)*( special.erf(var1_(B)) +  special.erf(var2_(B)) )   *\
                                                   np.log((1/2)*( special.erf(var11(B)) +  special.erf(var22(B)) ))


    func3= lambda B: (np.exp(-B*B/2)/np.sqrt(2*pi))   *   (1/2)*( special.erf(var1(B))   +  special.erf(var2(B)) )   *\
                                                   np.log((1/2)*( special.erf(var11_(B)) +  special.erf(var22_(B)) ))
                                                   
    func_test= lambda B: (func2(B)-func1(B))/dq
    func_test_=lambda B: (func3(B)-func1(B))/dq

    print(integrate.quad(func_test,-cut,cut)[0],integrate.quad(func_test_,-cut,cut)[0],integrate.quad(func_test,-cut,cut)[0]+integrate.quad(func_test_,-cut,cut)[0])

def State_evolution_original(x):
    Nt=int(x[0])
    q_in=x[1]
    m_in=x[2]
    κ_old=x[3]
    κ_new=x[4]
    α=x[5]
    
    P = lambda w: np.exp(-w*w/2)
    N_w=integrate.quad(P, -κ_old, κ_old)[0]
    
    m=np.zeros(Nt+1)
    q=np.zeros(Nt+1)
    m_hat=np.zeros(Nt+1)
    q_hat=np.zeros(Nt+1)
    
    xo=1
    η=0.01
    conv=0
    
    cut=8
    dcut=0.001
    
    ####### Derivatives channel out #######
    #######################################
      
    def dq_Ф_out_simple(Q,q,m,κ,α,cut):
        eps=10**(-40)
        
        def dZ_out(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=(-np.exp(-var1**2)+np.exp(-var2**2))/(np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
            
        def f(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=κ*(np.exp(-var1**2)+np.exp(-var2**2))/(2*abs(Q-q)*np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
             
        def potential_simple(x):
            var1=(+m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            var2=(-m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out1=m/(2*np.sqrt(q*(q-m**2)))*dZ_out(x)*(-np.exp(-var1**2)+np.exp(-var2**2))/np.sqrt(2*pi) 
            out2= (     (  x/(2*np.sqrt(q)) +np.sqrt(q)*x/(2*(Q-q)) )*dZ_out(x) + f(x)       ) * Z
            
            return (out1+out2)*(np.exp(-x*x/2)/N_w)
                

            
        return α*integrate.quad(potential_simple,-cut,cut)[0]
      
    def dm_Ф_out_simple(Q,q,m,κ,α,cut):
        eps=10**(-40)
           
        def dZ_out(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=(-np.exp(-var1**2)+np.exp(-var2**2))/(np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
  
        def potential_simple(x):
            var1=(+m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            var2=(-m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            
            out=-np.sqrt(q/(q-m**2))*dZ_out(x)*(-np.exp(-var1**2)+np.exp(-var2**2))/np.sqrt(2*pi) 
            
            return (out)*(np.exp(-x*x/2)/N_w)
            
    
        
        return α*integrate.quad(potential_simple, -cut, cut)[0]
      
    def potential_cut(x,Q,q,m,κ,α):
        eps=10**(-40)
        
        def dZ_out(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=(-np.exp(-var1**2)+np.exp(-var2**2))/(np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
            
            
        var1=(+m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
        var2=(-m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            
        out=-np.sqrt(q/(q-m**2))*dZ_out(x)*(-np.exp(-var1**2)+np.exp(-var2**2))/np.sqrt(2*pi) 
            
        return abs(out*(np.exp(-x*x/2)/N_w))
    
    ######## Derivatives channel in #######
    #######################################    
    def dqhat_Ф_in(q_hat,m_hat):
        eps=10**(-10)
        potential_1= lambda t: (1/(2*np.sqrt(q_hat+eps)))*t*np.tanh(m_hat*xo+np.sqrt(q_hat+eps)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    
    def dmhat_Ф_in(q_hat,m_hat):
        eps=10**(-10)
        potential_1= lambda t: xo*np.tanh(m_hat*xo+np.sqrt(q_hat+eps)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    
    
    
    
    
    ####### Running the SE #######
    ##############################     
    q[0]=q_in
    m[0]=m_in
    
    for t in range(Nt):
        
        ##### defining the cut in the integrals #####
        test1=0
        test2=0
        threshold=10**(-10)
        
        for j in range(4000):
            
            
            if abs(potential_cut(cut,1,q[t],m[t],κ_new,α))<threshold:
                test2=test1
                test1=+1
                cut+=-dcut 
                 
            if abs(potential_cut(cut,1,q[t],m[t],κ_new,α))>threshold:
                test2=test1
                test1=-1
                cut+=+dcut 
    
            if test1*test2<0:
                break
                
            if cut<0.01:
                cut=0.01
                break
                
            
            
        ##### hat variables #####
        q_hat[t]=-2*dq_Ф_out_simple(1,q[t],m[t],κ_new,α,cut) 
        m_hat[t]=dm_Ф_out_simple(1,q[t],m[t],κ_new,α,cut)
        

        ##### usuelle variables #####        
        if q_hat[t]==0.0 or m_hat[t]==0.0:
            print("!!! pb in integrals")
            q_new=max(0.9999*q[t],0)
            m_new=max(0.9999*m[t],0)
        else:
            q_new=(1-2*dqhat_Ф_in(q_hat[t],m_hat[t]))
            m_new=(dmhat_Ф_in(q_hat[t],m_hat[t]))
        q[t+1]=η*q_new+(1-η)*q[t]
        m[t+1]=η*m_new+(1-η)*m[t]
        
        
        
        
        print("t,α,κ_old,κ_new,η,cut:",t,α,round(κ_old,5),round(κ_new,5),η,round(cut,5))
        print("q,m,dq:",1,round(q[t],9),round(m[t],9),(abs(q[t]-q[t+1])/(1-q[t])))
        print("q_hat,m_hat,conv:",round(q_hat[t],9),round(m_hat[t],9),conv)
        print("test !!")
        if conv>30:
            test_ex(q[t],m[t],κ_old,κ_new,α,cut)
        print("")
        
        
        ### Criteria for convergence of SE ####
        if abs(q[t]-q[t+1])/(1-q[t])<1*10**(-4) and t>30:
            η=0.01
            conv+=1
            t_mem=t
            
        if abs(q[t]-q[t+1])/(1-q[t])<10**(-4) and conv>40:
            t_mem=t
            print("Converging normal")
            break
        
        if t==Nt-1:
            t_mem=t
            
        #### Criteria if SE explodes ####
        if (q_hat[t])>500 and t>10:
            t_mem=t
            print("Converging q_hat")
            break

        #### Criteria if SE has a problem ####
        if math.isnan(q[t+1])==True or math.isnan(m[t+1])==True:
            q[t+1]=max(0.9999*q[0],m[0]**2)
            m[t+1]=max(0.9999*m[0],0)
            
    x[1]=q[t_mem]
    x[2]=m[t_mem]
    return[q[t_mem],m[t_mem],q_hat[t_mem],m_hat[t_mem],cut]    

##################################
##################################











###############################################################
###### Distribution of interaction for the Giant cluster ######
###############################################################

def Distrib_interact_giant_cluster(α,κ,N):
    
    #### Parameter for the discretization of the distribution of interactions ####
    No=5
    No_tot=2*No
    discretization="rescaled"
    
    
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
    
    
    ξo=slope(κ)

    # A is the update matrix to go from P^{t-1}[w] to P^{t}[w] where we added the symmetry w→-w (u_j=→-u_j) to split 
    A=np.zeros((No_tot,No_tot))
    
    u_plus=np.zeros(No)
    u_minu=np.zeros(No)
    Δ_list=np.zeros(2*No)
    
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
    
    
    
    print("step 1: We compute the update matrix A")

    for i in range(No_tot):
         for j in range(No_tot):
     
            u_i=u_tot[i]
            u_j=u_tot[j]
            Δ=Δ_list[j]
         
            if j!=i+1 and j!=i:
                func= lambda v: (  e_(-(v+m*u_i)**2/(2*(1-m*m))) /sqt_(2*pi*(1-m*m))  )/H(κ,m*v,1-m*m) 
                A[i][j]=integrate.quad(func, u_j-Δ,u_j )[0]#(  e_(-(u_j+m*u_i)**2/(2*(1-m*m))) /sqt_(2*pi*(1-m*m))  )*Δ/H(κ,m*u_j,1-m*m)#
            if j==i+1:
                func= lambda v: (  e_(-(v+m*u_i)**2/(2*(1-m*m))) /sqt_(2*pi*(1-m*m))  )/H(κ,m*v,1-m*m) 
                A[i][j]=integrate.quad(func, u_j-Δ,u_j )[0]#(  e_(-(u_j+m*u_i)**2/(2*(1-m*m))) /sqt_(2*pi*(1-m*m))  )*Δ/H(κ,m*u_j,1-m*m) #
         
            if j==i:
                func= lambda v: (  e_(-(v+m*u_i)**2/(2*(1-m*m))) /sqt_(2*pi*(1-m*m))  )/H(κ,m*v,1-m*m) 
                A[i][j]=integrate.quad(func, u_j-Δ,u_j )[0]#(  e_(-(u_j+m*u_i)**2/(2*(1-m*m))) /sqt_(2*pi*(1-m*m))  )*Δ/H(κ,m*u_j,1-m*m)#

     
    print("step 2: We get the eigenvectors of A")        
    w,v=    np.linalg.eig(A) 
    P_x=u_tot
    
    index_max=np.argmax(w)    

    func_Gauss= lambda x: e_(-x*x/2)/sqt_(2*pi)
    Norm_Gauss=integrate.quad(func_Gauss,-κ,κ)[0]
    func_rescaling= lambda x: (e_(-x*x/2)/sqt_(2*pi))*np.interp(x ,P_x,abs(v[:,index_max]))
    Norm=integrate.quad(func_rescaling,-κ,κ)[0]    
    
    P_y_Gauss=func_Gauss(P_x)/Norm_Gauss
    P_y=func_rescaling(P_x)/Norm
    
    return P_x,P_y,P_y_Gauss
      
###############################################################
###############################################################










##############################################
###### CONNECTED-SOLUTION COMPUTATION   ###### 
##############################################

def Quenched_theory(N,κ_new,κ_old,α):
    ### initialization of the distribution
    N_x=300
    N_x_cut=120
    cut=0.95*κ_old
    P_x,P_y=distribution_initialization(κ_old,cut,N_x,N_x_cut)
    
    m=1-2/N
    Ntot=int(log_(0.1)/log_(m))
    
    for j in range(Ntot):
        
        if j%max(1,int(Ntot/100))==0:
            print("Quench theory, Size:",N)
            print('Completion:',round(100*j/Ntot,2),'%')
            

        if j!=0:
            κ_old=κ_new
                
        pot= potential(κ_new,κ_old,α,m ,P_x,P_y)
        if pot>0:
            cut=max(κ_new/2,κ_new-10*sqt_(2*(1-m*m)))
            P_x,P_y=distribution_update_2(κ_new,κ_old,α,m,P_x,P_y,cut,N_x,N_x_cut)
        else:
            break

        mo=(m**(j+1))
    return mo



###### Potential ######
def potential(κ_new,κ_old,α,m,P_x,P_y):
    
    P = lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(abs(w),P_x,P_y) 
    N_w=integrate.quad(P, -κ_old,κ_old)[0]
    
    entropic_term=-log_((1-m)/2)*(1-m)/2-log_((1+m)/2)*(1+m)/2
    
    var=  lambda w:(1/2)*erf_((κ_new-m*w)/sqt_(2*(1-m*m)))+(1/2)*erf_((κ_new+m*w)/sqt_(2*(1-m*m)))
    func= lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(abs(w),P_x,P_y) *log_(var(w))

    cut=max(0,κ_old-20*sqt_(2*(1-m*m)))
    
    energetic_term=(α/N_w)*integrate.quad(func, cut ,κ_old )[0]+(α/N_w)*integrate.quad(func,-κ_old,-cut )[0]+(α/N_w)*integrate.quad(func,-cut,cut )[0]

        
    return entropic_term+energetic_term


#### Solver for Adiabatic theory ####
def optimal_distance_potential_3(dκ,κ_old,α,m,P_x,P_y):
    
    def opening_gap(dκ):
        κ_new=κ_old+abs(dκ)
    
        out= (abs(potential(κ_new,κ_old,α,m,P_x,P_y))**(1/10))
        
        return out

    dκ_new=optimize.fsolve(opening_gap,dκ)[0]#,disp=False)[0]
    dκ_new=abs(dκ_new)
    if dκ_new>0.1:
        dκ_new=optimize.fmin(opening_gap,dκ/2,disp=False)[0]
        
    print('dκ,error:',dκ_new,opening_gap(dκ_new))
    
    return abs(dκ_new)


###### Update of the interaction distribution ######

def distribution_initialization(κ_ini,cut,N_x,N_x_cut):
    P_x=np.zeros(N_x)
    P_y=np.zeros(N_x)
    for k in range(N_x):
        if k<N_x_cut:
            P_x[k]=cut*(k/(N_x_cut))
            P_y[k]=1
        
        else:
            P_x[k]=cut+(κ_ini-cut)*((k-N_x_cut)/(N_x-N_x_cut))
            P_y[k]=1
    
    return P_x,P_y

def distribution_update_2(κ_new,κ_old,α,m,P_x,P_y,cut,N_x,N_x_cut):
    
    P_x_new=np.zeros(N_x)
    P_y_new=np.zeros(N_x)
    
    
    for k in range(N_x):
        
        if k<N_x_cut:
            P_x_new[k]=cut*(k/(N_x_cut))
            func_P=lambda v: (e_(-v*v/2)/sqt_(2*pi)) * np.heaviside(κ_old-abs(m*P_x_new[k] + sqt_(1-m*m)*v),1)       \
                                                     *          np.interp(abs(m*P_x_new[k] + sqt_(1-m*m)*v) ,P_x,P_y)\
                                                     /H(κ_new,   m*m*P_x_new[k]+m*sqt_(1-m*m)*v,   1-m*m)            ### !!!! take abs(x) to symetrize the distribution
            
            vo= (κ_old-m*P_x_new[k])/sqt_(1-m*m)        
            v1=(-κ_old-m*P_x_new[k])/sqt_(1-m*m)      

            x_min=max(-10,v1-0.001)
            x_max=min(vo+0.001,10) 
                     
            print_=0
            if print_==1:
                                                 
                N_test=300
                x_list=np.zeros(N_test)
                y_list=np.zeros(N_test)
                                                           
                for j in range(N_test):
                    x_list[j]=x_min+(x_max-x_min)*(j/N_test)
                    y_list[j]=func_P(x_list[j])
                                                 
                plt.plot(x_list,y_list,label='true')
                plt.legend()
                plt.show()    
                                                 
            P_y_new[k]=integrate.quad(func_P, x_min,x_max )[0]
            
        else:
            P_x_new[k]=cut+(κ_new-cut)*((k-N_x_cut)/(N_x-N_x_cut))
            func_P=lambda v: (e_(-v*v/2)/sqt_(2*pi)) * np.heaviside(κ_old-abs(m*P_x_new[k] + sqt_(1-m*m)*v),1)       \
                                                     *          np.interp(abs(m*P_x_new[k] + sqt_(1-m*m)*v) ,P_x,P_y)\
                                                     /H(κ_new,   m*m*P_x_new[k]+m*sqt_(1-m*m)*v,   1-m*m)            ### !!!! take abs(x) to symetrize the distribution
            
            
            vo= (κ_old-m*P_x_new[k])/sqt_(1-m*m)        
            v1=(-κ_old-m*P_x_new[k])/sqt_(1-m*m)      

            x_min=max(-10,v1-0.001)
            x_max=min(vo+0.001,10) 
            
            if 0.9<((k-N_x_cut)/(N_x-N_x_cut)):
                print_=0
            else:
                print_=0
                
                
                
            if print_==1:
                                                 
                N_test=300
                x_list=np.zeros(N_test)
                y_list=np.zeros(N_test)
                                    
                for j in range(N_test):
                    x_list[j]=x_min+(x_max+3.1*abs(x_max)-x_min)*(j/N_test)
                    y_list[j]=func_P(x_list[j])
                                           
                plt.plot([x_max,x_max],[0,max(y_list)])
                plt.plot(x_list,y_list,label='true')
                plt.legend()
                plt.show()    
                
            P_y_new[k]=integrate.quad(func_P,x_min ,x_max )[0]
            
    return P_x_new,P_y_new






#############################
###### Monte-Carlo sim ######
#############################

def Monte_Carlo(α,κ_ini,κ_max,Nκ,N,write,ticket):
    
    dκ=(κ_max-κ_ini)/Nκ
    
    β_eff=0           #Effective temp that forces you to decorrelate
    M=int(α*N)        #Number of interactions
    
    N_flip=95*N      #Number of trial flips per dκ
    N_flip_threshold=400
    
    flip_break=0
    
    
    
    perceptron=     lambda x,G,κ: np.heaviside(abs(G.dot(x))-κ,1)
    perceptron_sum= lambda x,G,κ:np.sum(perceptron(x,G,κ))
    
    perceptron_dif=     lambda interaction_vector,G_transpose,indice_flip,sign_flip,κ: np.heaviside(abs(interaction_vector[:]+2*sign_flip*G_transpose[:][indice_flip])-κ,1) 
    perceptron_sum_dif= lambda interaction_vector,G_transpose,indice_flip,sign_flip,κ: np.sum(perceptron_dif(interaction_vector,G_transpose,indice_flip,sign_flip,κ))


    accept_flip= lambda sign_flip,β: np.sign(    e_(-β*sign_flip)/(2*cosh_(β))      -   np.random.uniform(0,1,1)  )       
 
    
 
    
 
    
    
    ########## Generates G ##############
    #####################################
    xo=np.ones(N,dtype=int)
    G=np.zeros((M,N)) 
    u=np.zeros(M)
        
    print('Step 1: generating N*M Gaussian random numbers')
    G_o=np.random.normal(0, 1, size=(M,N))

    print('Step 2: subtracting the xo direction')            
    G_scalar_prod=G_o.dot(xo)   
    G=G_o-np.outer(G_scalar_prod,xo)/N
    
    
    print('Step 3: generating the interactions in the xo direction')   
    #N_test=5000
    #u=np.random.normal(0, 1, size=(N_test,M))
    
    u=np.zeros(M)
    for i in range(M):
        for test in range(500000):
            u[i]=np.random.normal(0, 1, 1)
            
            if abs(u[i])<κ_ini:
                break
            if test==500000-2:
                print('problem!!!!!!')
            
            
        G[i,:]+=u[i]*xo[:]/np.sqrt(N)
        
    G=G/np.sqrt(N)
    G_transpose=np.transpose(G)
    
    #####################################
    #####################################
    
    
    
    
    
    ####################################
    ### initialization of the system ###
    ####################################
    x=xo+0.00
    mag=1
    
    interaction_vector=G.dot(x)
    indice_flip=0
    ####################################
    ####################################
    
    
    
    ########################################
    ### Observables for the correlations ###
    ########################################    
    N_mem=150
    x_mem=np.zeros((N_mem,N))
    
    
    dk_mem=(κ_max-κ_ini)/N_mem
    indice_mem=0
    kappa_list_mem=np.zeros(N_mem)
    
    
    κ=κ_ini+0 # Important to keep the +0 !!
    dκ_tot=0
    for  k in range(Nκ):
        κ+=dκ
        dκ_tot+=dκ
        if dκ_tot>dk_mem:
            kappa_list_mem[indice_mem]=κ
            indice_mem+=1
            dκ_tot=0
            
    for k in range(N_mem-indice_mem):
        kappa_list_mem[indice_mem+k]= kappa_list_mem[indice_mem-1]\
                                      +(κ_max-kappa_list_mem[indice_mem-1])*(k+1)/(N_mem-indice_mem)
                                      
        kappa_list_mem[indice_mem+k]=min(kappa_list_mem[indice_mem+k],κ_max)
        

    indice_mem=0
    dκ_tot=0
    ######################################## 
    ######################################## 


    if write==1:      
        file=open('Monte_Carlo_magnetization_'+str(ticket)+'(alpha='+str(α)+' N='+str(N)+' dkappa='+str(dκ)+').txt','w')   
        file.write('kappa	magnetization')
        file.write("\n")
        
        file2=open('Monte_Carlo_correlation_'+str(ticket)+'(alpha='+str(α)+' N='+str(N)+' dkappa='+str(dκ)+').txt','w')   
        file2.write('kappa	(x/y)')
        file2.write("\n")
        
        file2.write('		') 
        for k in range(N_mem):
            file2.write(str(kappa_list_mem[k]))
            file2.write('		')   
        file2.write("\n")            
        
        
    
    
    mag_list=np.ones(Nκ)
    flip_list=np.zeros(Nκ)
    kappa_list=np.zeros(Nκ)
    mag_mem=1
    
    for k in range(Nκ):
        κ=κ_ini+(k+1)*dκ
        kappa_list[k]=κ
        
        
        dκ_tot+=dκ
        if κ>=kappa_list_mem[indice_mem]:
            x_mem[indice_mem][:]=x[:]
            
            if write==1:
                file2.write(str(kappa_list_mem[indice_mem]))
                file2.write('		')   
                for j in range(indice_mem+1):
                    file2.write(   str(x_mem[indice_mem][:].dot(x_mem[j][:])/N)       )
                    file2.write('		')   
                file2.write("\n")   
                
            indice_mem+=1
        
        
        
        count_break=0
        
        for j in range(N_flip):
            
            
            ##### Select an indice #####
            indice_flip+=1
            indice_flip=int(indice_flip%N) 
            sign_flip=-x[indice_flip]
            
            
            ##### Flipping rule #####
            if flip_break==0 and accept_flip(sign_flip,β_eff)>0:
                   
                if perceptron_sum_dif(interaction_vector,G_transpose,indice_flip,sign_flip,κ)<1/M:
         
                    x[indice_flip]*=-1
                    interaction_vector[:]=interaction_vector[:]+2*sign_flip*G_transpose[:][indice_flip]
                    
                    mag=mag+2*sign_flip/N
                    mag_mem=min(mag,mag_mem)
                    
                    flip_list[k]+=sign_flip
                    if sign_flip<0:
                        count_break+=1
                        


            #### Breaking rules ####                    
            if flip_break==0 and mag_list[k-1]<0:
                flip_break=1
                 
            if count_break>N_flip_threshold:
                break
            
            if j>N/2 and mag_mem>1-3/N: #### This allows to go fast at the beginning as very few flips can be done for all lot of kappa's
                break
            
            
            
            
        #### Update mag ####
        mag_list[k]=mag_mem
        if write==1:
            file.write(str(κ))
            file.write('		')   
            file.write(str(mag_list[k]))
            file.write("\n")   
            
            
        print('N:',N)
        print('k,Nκ:',k,Nκ)
        print('κ,κ_ini:',κ,κ_ini)
        print('mag:',mag_list[k],', Nb flip:',round(N*(1-mag_list[k])/2,0))
        print("")
                

    if write==1:
        file.close()
        file2.close()
        
        
    plt.plot(kappa_list,log_(1-mag_list+1/N)/log_(10))
    plt.title(r'$\alpha=$'+str(α)+'   '+r'$\kappa_{initial}=$'+str(κ_ini)+'   '+r'$d\kappa=$'+str(dκ))
    plt.ylabel(r'$m(t,0)$')
    plt.xlabel(r'$\kappa$')
    plt.show()
    
    plt.plot(kappa_list,mag_list+1/N)
    plt.title(r'$\alpha=$'+str(α)+'   '+r'$\kappa_{initial}=$'+str(κ_ini)+'   '+r'$d\kappa=$'+str(dκ))
    plt.ylabel(r'$m(t,0)$')
    plt.xlabel(r'$\kappa$')
    plt.show()

    plt.plot(kappa_list,flip_list)
    plt.title(r'$\alpha=$'+str(α)+'   '+r'$\kappa_{initial}=$'+str(κ_ini)+'   '+r'$d\kappa=$'+str(dκ))
    plt.ylabel(r'$N_{flip}$')
    plt.xlabel(r'$\kappa$')
    plt.show()

def Monte_Carlo_2(α,κ_old_list,κ_new_list,N_list,mo_list,write,ticket):
    N_try=len(κ_new_list)

    
    perceptron=     lambda x,G,κ: np.heaviside(abs(G.dot(x))-κ,1)
    perceptron_sum= lambda x,G,κ:np.sum(perceptron(x,G,κ))
    
    perceptron_dif=     lambda interaction_vector,G_transpose,indice_flip,sign_flip,κ: np.heaviside(abs(interaction_vector[:]+2*sign_flip*G_transpose[:][indice_flip])-κ,1) 
    perceptron_sum_dif= lambda interaction_vector,G_transpose,indice_flip,sign_flip,κ: np.sum(perceptron_dif(interaction_vector,G_transpose,indice_flip,sign_flip,κ))


    accept_flip= lambda sign_flip,β: np.sign(    e_(-β*sign_flip)/(2*cosh_(β))      -   np.random.uniform(0,1,1)  )       
 
    
 


    if write==1:      
        file=open('Monte_Carlo_magnetization_'+str(ticket)+'(alpha='+str(α)+').txt','w')   
        file.write('Trial_number	N	kappa_old	kappa_new	time	time_rescaled	magnetization	magnetization_theoric_final')
        file.write("\n")
        
        
    
    ####################################
    ### magnetization  of the system ###
    ####################################
    N_mem=150                                    #Number of configuration stored
    mag_list_time=np.zeros((N_try,N_mem))        #Store the overlap with the planted signal in rescaled time steps
    time_list_mem=np.zeros((N_try,N_mem))        #Store the true time steps of the quench
    time_cut=np.zeros(N_try)
    ####################################
    ####################################
        
        
    
    for k in range(N_try):
        κ=κ_new_list[k]
        κ_old=κ_old_list[k]
        N=int(N_list[k])
        
        β_eff=0           #Effective temp that forces you to decorrelate
        M=int(α*N)        #Number of interactions
        
        N_flip=int(5*N*N)    #Number of trial flips per try

    
        
        
    
        ########## Generates G ##############
        #####################################
        xo=np.ones(N,dtype=int)
        G=np.zeros((M,N)) 
        u=np.zeros(M)
        
        print('Step 1: generating N*M Gaussian random numbers')
        G_o=np.random.normal(0, 1, size=(M,N))
        #G_o=np.random.uniform(low=-sqt_(3), high=sqt_(3), size=(M,N))

        print('Step 2: subtracting the xo direction')            
        G_scalar_prod=G_o.dot(xo)   
        G=G_o-np.outer(G_scalar_prod,xo)/N
    
    
        print('Step 3: generating the interactions in the xo direction')   
        u=np.zeros(M)
        for i in range(M):
            for test in range(50000000):
                u[i]=np.random.normal(0, 1, 1)
            
                if abs(u[i])<κ_old:
                    break
                if test==50000000-2:
                    print('problem!!!!!!')
            
            
            G[i,:]+=u[i]*xo[:]/np.sqrt(N)
        
        G=G/np.sqrt(N)
        G_transpose=np.transpose(G)
        
        #####################################
        #####################################        
        
        
        ####################################
        ### initialization of the system ###
        ####################################
        x=xo+0.00
        mag_mem=1
        mag=1
        
        interaction_vector=G.dot(x)
        indice_flip=0
        ####################################
        ####################################
        
        
        
        #########################################################
        ### Observables for the correlations and interactions ###
        #########################################################    
        x_mem=np.zeros((N_mem,N))
        
        dflip_mem=int(N/1000) # We store a new config. each time the system has flipped "dflip_mem" spins 
        flip_count=0
        
        correlation_list_mem=np.zeros((N_mem,N_mem)) # Store the correlations in rescaled time steps
        correlation_list_theory=np.zeros(N_mem)      # Theoretical correlations in rescaled time steps
        int_list=np.array([])                        # Store the interactions
        
        for l in range(N_mem):
            correlation_list_theory[l]=(1-2*dflip_mem/N)**l
            correlation_list_mem[l][l]=1.0
            
            
        x_mem[0][:]=x[:]+0.00
        
        time_list_mem[k][0]=0
        mag_list_time[k][0]=1
        correlation_list_mem[0][0]=x_mem[0][:].dot(x_mem[0][:])/N
        indice_mem=1
        ######################################## 
        ########################################   
        

        ### Breaking Variables (Not used for quench) ###
        count_break=0
        flip_break=0
        ################################################
        

        
        for j in range(N_flip):
            
            if indice_mem==N_mem:
                break

            
            #######################################################
            ###### Check if we have make dflip_mem spin flips######
            #######################################################

            if flip_count>dflip_mem and indice_mem<N_mem:
                
                #### Print the completion ####
                if indice_mem%(max(int(N_mem/100),1))==0:
                    print('Try number:',k,'N=',N)
                    print(round(100*j/N_flip,2),'% completion, m(t,0)=',mag,round(100*indice_mem/N_mem),'% memory taken')
                    print("")

                
                time_cut[k]=indice_mem
                
                ###### Computing the rescaled time ###### 
                time_list_mem[k][indice_mem]=j
                #########################################
                
                
                ###### Computing the overlap with planted conf. ######
                mag_list_time[k][indice_mem]=mag
                ######################################################
                
                ###### Computing the correlations ######
                x_mem[indice_mem][:]=x[:]+0.00
                
                for l in range(indice_mem):
                    correlation_list_mem[indice_mem][l]=x_mem[indice_mem][:].dot(x_mem[l][:])/N
                ########################################
                
                
                ###### Computing the interactions ###### 
                if mag<10/N:
                    int_list=np.append(int_list,G.dot(x),axis=0)
                ########################################                    
                               
                indice_mem+=1
                flip_count=0
                
                
                if write==1:
                    file.write(str(k))
                    file.write('		')  
                    file.write(str(N))
                    file.write('		') 
                    file.write(str(κ_old))
                    file.write('		') 
                    file.write(str(κ))
                    file.write('		')  
                    file.write(str(time_list_mem[k][indice_mem-1]))
                    file.write('		') 
                    file.write(str(indice_mem-1))
                    file.write('		') 
                    file.write(str(mag_list_time[k][indice_mem-1]))
                    file.write('		') 
                    file.write(str(mo_list[k]))
                    file.write("\n")  
                    
                    
            #######################################################
            #######################################################
            #######################################################

                
            
            
            #######################################################
            ######               The Dynamics                ######
            #######################################################            
    
            
            ##### Select an indice #####
            indice_flip+=1
            indice_flip=int(indice_flip%N) 
            sign_flip=-x[indice_flip]
            
            
            ##### Flipping rule #####
            if flip_break==0:# and accept_flip(sign_flip,β_eff)>0:
                   
                if perceptron_sum_dif(interaction_vector,G_transpose,indice_flip,sign_flip,κ)<1/M:
         
                    x[indice_flip]*=-1
                    interaction_vector[:]=interaction_vector[:]+2*sign_flip*G_transpose[:][indice_flip]
                    
                    mag=mag+2*sign_flip/N
                    mag_mem=min(mag,mag_mem)
                    
                    flip_count+=1
                    
                else:
                    
                    flip_count+=0
                    
             
        
            else:
                flip_count+=0
                count_break+=1


            #### Breaking rules ####                    
            #if flip_break==0 and mag_list[k-1]<0:
            #    flip_break=1
                 

            #if j>N/2 and mag_mem>1-3/N: #### This allows to go fast at the beginning as very few flips can be done for all lot of kappa's
            #    break
            
            #if count_break>100*N:
            #    time_cut[k]=j
            #    break
            
            #######################################################
            #######################################################
            #######################################################


            
 
        if ticket>0:

            ### printing overlap with the planted conf. ###
            m_fin=0.01
            index=int(log_(m_fin)/(log_(abs(1-2*dflip_mem/N))))
            
            #plt.plot(log_(1-mag_list_time[k][0:min(index,indice_mem)]+1/N)/log_(10))
            #plt.title(r'$\alpha=$'+str(α)+'   '+r'$\kappa_{initial}=$'+str(κ_ini))
            #plt.ylabel(r'$\log_{10}[1-m_{j,0}]$')
            #plt.xlabel(r'$j$')
            #plt.show()
    
            plt.plot(mag_list_time[k][0:min(index,indice_mem)])
            plt.plot(correlation_list_theory[0:min(index,indice_mem)])
            plt.title(r'$\alpha=$'+str(α)+'   '+r'$\kappa_{initial}=$'+str(κ_old))
            plt.ylabel(r'$m_{j,0}$')
            plt.xlabel(r'$j$')
            plt.show()
        
        if ticket>5:
            # Normal time
            ### printing correlations ###
            N_plot=9
            plot_index=0
            
            m_fin=0.01
            index=int(log_(m_fin)/log_((1-2*dflip_mem/N)))
            
            for j in range(indice_mem):
                
                if int(j%20)==1 and plot_index<=N_plot:
                    plot_index+=1
                    
                    time_list_new=np.zeros(min(indice_mem-j,index))
                    correlation_new=np.zeros(min(indice_mem-j,index))
                    
                    for l in range(min(indice_mem-j,index)):
                        time_list_new[l]  =abs(time_list_mem[k][j+l]-time_list_mem[k][j])
                        correlation_new[l]=correlation_list_mem[j+l][j]
                        
                    plt.plot(time_list_new,correlation_new,label="j'="+str(int(time_list_mem[k][j])))
        
    
            plt.ylabel(r"$m_{j,j'}$")
            plt.xlabel(r"$\vert j-j'\vert $")
            plt.title(r'$\alpha=$'+str(α)+r'$\quad\kappa=$'+str(κ)+r'$\quad N=$'+str(N))
            plt.legend()
            plt.show()
            
            
            # Rescaled time
            ### printing correlations ###
            N_plot=9
            plot_index=0
            
            m_fin=0.01
            index=int(log_(m_fin)/log_((1-2*dflip_mem/N)))

            for j in range(indice_mem):
                
                if int(j%20)==1 and plot_index<=N_plot:
                    plot_index+=1
                    
                    correlation_new=np.zeros(min(indice_mem-j,index))
                    for l in range(min(indice_mem-j,index)):
                        correlation_new[l]=correlation_list_mem[l+j][j]
                    
                    plt.plot(correlation_new,label='j='+str(int(time_list_mem[k][j])))
        
    
            plt.plot(correlation_list_theory,c='black',linestyle='--',label='theoretical prediction')    
            plt.ylabel(r"$m_{j,j'}$")
            plt.xlabel(r"$\vert j-j'\vert $")
            plt.xlim([0,index])
            plt.title(r'$\alpha=$'+str(α)+r'$\quad\kappa=$'+str(κ)+r'$\quad N=$'+str(N))
            plt.legend()
            plt.show()
    
        
            ### printing interaction distribution ###
            P_x,P_y,P_y_Gauss=Distrib_interact_giant_cluster(α,κ,N)
        
        
            func_Gauss= lambda x: e_(-x*x/2)/sqt_(2*pi)
            Norm_Gauss=integrate.quad(func_Gauss,-κ,κ)[0]
        
            κ_cut=-κ+3*sqt_(1-(1-2/N)**2)
            bin_tot=int(100*2*κ/abs(-κ-κ_cut))
            
            plt.plot(P_x,P_y,c='black',linestyle='--')
            plt.plot(P_x,P_y_Gauss,c='grey',linestyle='dotted')
            plt.hist(int_list,bins=bin_tot,density=True)
            plt.show()
        
    
            plt.plot(P_x,P_y,c='black',linestyle='--')
            plt.plot(P_x,P_y_Gauss,c='grey',linestyle='dotted')
            plt.hist(int_list,bins=bin_tot,density=True)
            plt.xlim([-κ,κ_cut])
            plt.ylim([0,func_Gauss(0.95*κ_cut)/Norm_Gauss])
            plt.show()            
            
                
    
    
    if write==1:
        file.close()
        
    average_mag_list=np.zeros((N_try,N_mem))
    error_mag_list=np.zeros((N_try,N_mem))
    time_cut_list=np.zeros((N_try))+10**9
    
    
    N_limit=50
    average_mag_list_limited=np.zeros((N_try,N_limit))
    error_mag_list_limited=np.zeros((N_try,N_limit))
    time_list_new=np.zeros((N_try,N_limit))



    ### Computing the averages ###
    No=N_list[0]
    index=0
    number_avg=0
    
    for k in range(N_try):
        
        if N_list[k]==No:
            number_avg+=1
            average_mag_list[index][:]+=mag_list_time[k][:]
            time_cut_list[index]=min(time_cut_list[index],time_cut[k])
            
        else:
            average_mag_list[index][:]*=(1/number_avg)
            
            number_avg=1
            index+=1
            No=N_list[k]
            
            average_mag_list[index][:]+=mag_list_time[k][:]
            time_cut_list[index]=min(time_cut_list[index],time_cut[k])
            
    
    average_mag_list[index][:]*=(1/(number_avg))        
    ##############################
    
    
    ### Computing the variances ###    
    No=N_list[0]
    index=0


    for k in range(N_try):
        
        if N_list[k]==No:
            error_mag_list[index][:]=np.maximum(error_mag_list[index][:],abs(average_mag_list[index][:]-mag_list_time[k][:]))
           
        else:  
            index+=1
            No=N_list[k]    
            
            error_mag_list[index][:]=np.maximum(error_mag_list[index][:],abs(average_mag_list[index][:]-mag_list_time[k][:])) 
    ###############################

    for k in range(index+1):
        index_=-1
        for j in range(N_mem):
            if j%(max(1,int(N_mem/N_limit)))==0 and index_<N_limit-1:
                index_+=1
                
                average_mag_list_limited[k][index_]=average_mag_list[k][j]
                error_mag_list_limited[k][index_]  =error_mag_list[k][j]
                time_list_new[k][index_]=j
                
        if index_<N_limit-1:
            for j in range(N_limit-1-index_):
                average_mag_list_limited[k][j+1+index_]=average_mag_list_limited[k][index_]
                error_mag_list_limited[k][j+1+index_]  =error_mag_list_limited[k][index_]
                time_list_new[k][j+1+index_]           =time_list_new[k][index_]
                
                
        
    for k in range(index+1):
        plt.errorbar(time_list_new[k][:], average_mag_list_limited[k][:], yerr=error_mag_list_limited[k][:], fmt='-o',markersize=1)
    plt.plot(correlation_list_theory[0:indice_mem],c='black',linestyle='--')
    
    
    
    y_list=np.maximum(np.zeros(N_try)+correlation_list_theory[indice_mem-1],mo_list)
    x_list=np.zeros(N_try)
    for k in range(N_try):
        N=int(N_list[k])
        dflip_mem=int(N/250) # We store a new config. each time the system has flipped "dflip_mem" spins
        if mo_list[k]>correlation_list_theory[indice_mem-1]:
            x_list[k]=int(log_(mo_list[k])/(log_(abs(1-2*dflip_mem/N))))
        else:
            x_list[k]=N_mem-1
            
        
    plt.scatter(x_list,y_list,c='black')
    plt.show()
    
    return mag_list_time

#############################
#############################










α=0.5
κ_SAT=κ_SAT_finding(α)[0]
κ_max=2.5
Nκ=3500


Original_SE_loop=0
if Original_SE_loop==1:
    κ_min=1.05*κ_SAT
    κ_old=κ_SAT+0
    
    Running_original_state_evolution(α,κ_old,κ_max,κ_min,Nκ,write=1)


Monte_Carlo_loop=2



#### Adiabatic MC ####
if Monte_Carlo_loop==1:
    N=int(10**4.7)
    Monte_Carlo(α,κ_SAT,κ_max,Nκ,N,write=1,ticket=666)
    
    
    
#### Quenched MC (fixed kappa)####
if Monte_Carlo_loop==2:
    
    κ_new=1.
    κ_old=κ_SAT
    
    try_size=6
    N_true_list=[7000,10000,20000,40000,70000,100000]
    try_disorder_realization=10
    
    N_try=int(try_size*try_disorder_realization)
    
    
    κ_old_list=np.zeros(N_try)
    κ_new_list=np.zeros(N_try)
    N_list=np.zeros(N_try)
    mo_list=np.zeros(N_try)
    
    for i in range(try_size):
        mo=Quenched_theory(N_true_list[i],κ_new,κ_old,α)
        
        for j in range(try_disorder_realization): 

            κ_old_list[i*try_disorder_realization+j]=κ_old
            κ_new_list[i*try_disorder_realization+j]=κ_new
            N_list[i*try_disorder_realization+j]    =N_true_list[i]
            mo_list[i*try_disorder_realization+j]   =mo

    Monte_Carlo_2(α,κ_old_list,κ_new_list,N_list,mo_list,write=1,ticket=-5)
    


  

Connected_solution_loop=0
Write_connected_solution=0


#### This loops compute the adiabatic increase of the margin
if Connected_solution_loop==1:
    ##### Initialization #####
    ##########################
    κ_old=κ_ini
    dκ_guess=0.01

    ### optimal mag.
    N=10**(4)

    m=1-2/N
    dm =log_(1-m)/log_(10)
    Ntot=int(log_(0.01)/log_(m))

    dmo=dm

    method=3
    if method==2:
        phase=1
    
    if method==3:
        breaking_loop=0
        
        

    ### initialization of the distribution
    N_x=400
    N_x_cut=200
    cut=0.95*κ_ini
    P_x,P_y=distribution_initialization(κ_ini,cut,N_x,N_x_cut)
    
   
    mo_list=np.zeros(Ntot)      
    m_list=np.zeros(Ntot)      
    κ_list=np.zeros(Ntot)  
    norm=np.zeros(Ntot) 


    if Write_connected_solution==1:
        file=open('Connected_solution_'+str(method)+'(alpha='+str(α)+' dm='+str(dm)+').txt','w')   
        file.write('kappa_ini	kappa_old	kappa_new	dkappa	dm	dm_saddle	P_x	P_y')
        file.write("\n")
    
        file2=open('Connected_solution_simple_'+str(method)+'(alpha='+str(α)+' dm='+str(dm)+').txt','w')   
        file2.write('kappa_ini	kappa_old	kappa_new	dkappa	dm	dm_saddle')
        file2.write("\n")



    for k in range(Ntot):
    
    
        ######################################    
        ##### Get the correct new margin #####
        ######################################
        print('Adiabatic method')
        if method==2 or breaking_loop<10:
            print('iteration',k,'totale',Ntot)
            print('')
            print("Step 0 (optimal margin)")
            if method==2:
                print('phase:',phase)
            print("κ_old,dm:",κ_old,dm)
        
        if method==2: ## This method looks at the increment of margin κ for which the potential is fully increasing until a certain distance
    
            if phase==1:
                dκ=optimal_distance_potential_phase1(dκ_guess,κ_old,α,m,P_x,P_y)
                κ_new=dκ+κ_old
            
                dmo=optimal_distance_potential_phase1_test(dm+1.3,κ_new,κ_old,α,P_x,P_y)
            
            if phase==2:
                dκ,dmo=optimal_distance_potential_phase2(dκ_guess,dmo,κ_old,α,P_x,P_y)
                κ_new=dκ+κ_old
        
            if dmo-dm>0.3 and phase==1:
                dmo=(dmo+dm)/2
                phase=2   
                #break
    
        if method==3: ## This method looks at the increment of margin κ for which there is no OGP at a given distance
            if breaking_loop<10:
                dκ=optimal_distance_potential_3(dκ_guess,κ_old,α,m,P_x,P_y)
                κ_new=dκ+κ_old
                    
                if k>1:
                    print('breaking:',breaking_loop,dκ*(Ntot-k),abs(κ_list[0]-κ_list[1])*10**-4)
        
                if k>1 and dκ*(Ntot-k)<abs(κ_list[0]-κ_list[1])*10**-4:
                    breaking_loop+=1
            
        if method==3 and breaking_loop>=10:
            break
        
        
        ######################################
        ######################################
    
    
    
        ##############################################
        ##### Get the update of the distribution #####
        ##############################################
        print("")
        print("Step 1 (update distribution)")
        mo=1-10**(-abs(dmo))
        cut=max(κ_old/2,κ_old-20*sqt_(2*(1-mo*mo)))
    
        if method==2 or breaking_loop<10:
            P_x_new,P_y_new=distribution_update_2(κ_new,κ_old,α,m,P_x,P_y,cut,N_x,N_x_cut)
    
            func_P    =lambda x: np.interp(abs(x),P_x,P_y) 
            func_P_new=lambda x: np.interp(abs(x),P_x_new,P_y_new) 
    
            P_old = lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(abs(w),P_x,P_y) 
            N_w_old=integrate.quad(P_old, -κ_old,κ_old)[0]
    
            P_new = lambda w: (e_(-w*w/2)/sqt_(2*pi))*np.interp(abs(w),P_x_new,P_y_new) 
            N_w=integrate.quad(P_new, -κ_new,κ_new)[0]
    
        ##############################################
        ##############################################
    


        ##############################################
        ########## Print some stuff to help ##########
        ##############################################    
        if k<=4000000:
            print_=0
        else:
            print_=1
        
        if print_==1:
            print("")
            print("Step 2 (printing curves)")
    
            m_min=1-10**(-abs(dm)+1.5)
            m_max=1-10**(-abs(dm)-1.5)
            m_o=1-10**(-abs(dmo))
            order=-1
            dm_list,pot,pot_0,pot_1,pot_2=draw_pot(κ_old+dκ,κ_old,α,m_o,m_min,m_max,P_x,P_y,order)

            plt.plot(dm_list[0:len(dm_list)-2],pot[0:len(dm_list)-2],label=r'$d\kappa$')
            if order>-0.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_0[0:len(dm_list)-2],label=r'$d\kappa\neq 0$'+' 0',linestyle='--')
            if order>0.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_1[0:len(dm_list)-2],label='1',linestyle='--')
            if order>1.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_2[0:len(dm_list)-2],label='2',linestyle='--')
            
            order=-1
            dm_list,pot,pot_0,pot_1,pot_2=draw_pot(κ_old+1.2*dκ,κ_old,α,m_o,m_min,m_max,P_x,P_y,order)

            plt.plot(dm_list[0:len(dm_list)-2],pot[0:len(dm_list)-2],label=r'$2d\kappa$')
            if order>-0.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_0[0:len(dm_list)-2],label=r'$d\kappa\neq 0$'+' 0',linestyle='--')
            if order>0.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_1[0:len(dm_list)-2],label='1',linestyle='--')
            if order>1.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_2[0:len(dm_list)-2],label='2',linestyle='--')

        

            dm_list,pot,pot_0,pot_1,pot_2=draw_pot(κ_old+0.8*dκ,κ_old,α,m_o,m_min,m_max,P_x,P_y,order)

            plt.plot(dm_list[0:len(dm_list)-2],pot[0:len(dm_list)-2],label=r'$d\kappa/2$')
            if order>-0.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_0[0:len(dm_list)-2],label=r'$d\kappa\neq 0$'+' 0',linestyle='--')
            if order>0.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_1[0:len(dm_list)-2],label='1',linestyle='--')
            if order>1.5:
                plt.plot(dm_list[0:len(dm_list)-2],pot_2[0:len(dm_list)-2],label='2',linestyle='--')
            
            
            #plt.ylim([min(pot)/3,max(pot)])
            plt.plot([log_(1-m)/log_(10),log_(1-m)/log_(10)],[min(pot)/2,max(pot)*4/5])
           # plt.plot([-dmo,-dmo],[min(pot)/2,max(pot)*4/5])

            plt.title(r'$\kappa_{\rm old}=$'+str(κ_old)+"  "+r'$d\kappa=$'+str(dκ))
            if method!=3.1:
                plt.xlabel(r'$\log_{10}(1-m)$'+'  '+str(m**k))
            else:
                plt.xlabel(r'$\log_{10}(1-m)$'+'  '+str(mo))
            plt.ylabel('potential')
            plt.legend()
            plt.show()

        ##############################################
        ##############################################
    
    
    
        if Write_connected_solution==1:
            file.write(str(float(κ_ini)))
            file.write('		')
            file.write(str(float(κ_old)))
            file.write('		')
            file.write(str(float(κ_new)))
            file.write('		')
            file.write(str(dκ))
            file.write('		')
            file.write(str(dm))
            file.write('		')
            file.write(str(dmo))
            file.write('		')
            for j in range(N_x):
                file.write(str(P_x[j]))
                file.write('		')
            for j in range(N_x):
                file.write(str(P_y[j]))
                file.write('		')
            file.write("\n")
    
    
            file2.write(str(float(κ_ini)))
            file2.write('		')
            file2.write(str(float(κ_old)))
            file2.write('		')
            file2.write(str(float(κ_new)))
            file2.write('		')
            file2.write(str(dκ))
            file2.write('		')
            file2.write(str(dm))
            file2.write('		')
            file2.write(str(dmo))
            file2.write("\n")
        
        
        
        P_x=P_x_new
        P_y=P_y_new
       
        κ_old=κ_new
        κ_list[k]=κ_new
        dκ_guess=dκ
    
        m_list[k]=m   
        mo_list[k]=1
        for j in range(k+1):
            mo_list[k]=m_list[j]*mo_list[k]
            
        if method==3.1:
            mo_list[k]=1-(1-m)*k
            
        if method==2 or breaking_loop<10:
            print('')
            print('m(t,t-1):',m,'m(t,0):',mo_list[k])
            print('')
            print('')
            print('')
        
    
    if Write_connected_solution==1:
        file.close()
        file2.close()
   



#plt.plot(κ_list,log_(1-m_list)/log_(10),label=r'$m=\langle {\bf x}_{t-dt}{\bf x}_{t}\rangle$')
#plt.plot(κ_list,log_(1-mo_list)/log_(10),label=r'$m=\langle {\bf x}_{0}{\bf x}_{t}\rangle$')
#plt.title(r'$\alpha=$'+str(α)+'   '+r'$\kappa_{initial}=$'+str(κ_ini)+'   '+r'$d\kappa=$'+str(dκ))
#plt.xlabel(r'$\kappa$')
#plt.ylabel(r'$log_{10}(1-mm)$')
#plt.legend()
#plt.show()

#plt.plot(κ_list,m_list,label=r'$m=\langle {\bf x}_{t-dt}{\bf x}_{t}\rangle$')
#plt.plot(κ_list,mo_list,label=r'$m=\langle {\bf x}_{0}{\bf x}_{t}\rangle$')
#plt.title(r'$\alpha=$'+str(α)+'   '+r'$\kappa_{initial}=$'+str(κ_ini)+'   '+r'$d\kappa=$'+str(dκ))
#plt.xlabel(r'$\kappa$')
#plt.ylabel(r'$m$')
#plt.legend()
#plt.show()

import numpy as np
from scipy import stats
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import math 
from multiprocessing.pool import ThreadPool as Pool
pi=np.arccos(-1)


###### Computing κ_SAT ######
#############################
def κ_SAT_finding(α):
    def α_critical(κ_):
        κ=abs(κ_)
        integral=integrate.quad(lambda z: (np.exp(-z*z/2)/np.sqrt(2*pi)), -κ, κ)[0]
        out= α+np.log(2)/np.log(integral)
        #print("in (Computing κ_SAT):",κ)
        #print("out:",out)
        #print("")
        return  abs(out)
    
    κ_SAT=abs(optimize.fmin(α_critical,0.4))
    return κ_SAT


###### Computing Free energy and saddles ######
###############################################
def Free_energy_RS(q,q_hat,κ_new,α):
    
    def Ф_out(Q,q,κ,α):
        eps=10**(-45)
        Gab=q
        Q=1

        def potential_1(t):
            Z1= (1/2)*special.erf(  (κ-t*np.sqrt(Gab))/np.sqrt(2*(Q-q))  ) + (1/2)*special.erf(  (κ+t*np.sqrt(Gab))/np.sqrt(2*(Q-q))  )  
            Z=Z1*np.heaviside(abs(Z1)-eps,1)+eps*(1-np.heaviside(abs(Z1)-eps,1))
            return np.log(Z)
        
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -20, 20)[0]
    G_out=Ф_out(1,q,κ_new,α)
    
    def Ф_in(q_hat):
        eps=10**(-25)
        potential_1= lambda t: np.log(2*np.cosh(np.sqrt(q_hat+eps)*t) )
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2,-20,20)[0]
    G_in=Ф_in(q_hat)
    
    
    return 0.5*(1-q)*q_hat-α*G_out-G_in


def Free_energy_1RSB(q1,q1_hat,qo,qo_hat,m_,κ,α):
    
    def Ф_out(Q,q1,qo,m,κ):
        
        ѱ_out_rs= lambda x: np.log(      0.5*special.erf( (κ-x)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+x)/np.sqrt(2*(1-q1)) )        )
        ѱ_out_rs_int= lambda t: np.log(     integrate.quad(lambda z: (np.exp(-z*z/2)/np.sqrt(2*pi))*np.exp(m_*ѱ_out_rs(np.sqrt(q1-qo)*z+np.sqrt(qo)*t)), -20, 20)[0]      )
        return    integrate.quad(lambda t: (np.exp(-t*t/2)/np.sqrt(2*pi))*ѱ_out_rs_int(t), -20, 20)[0]      /m_
    
    G_out=Ф_out(1,q1,qo,m_,κ)
    
    
    
    
    def Ф_in(q1_hat,qo_hat,m_):
        
        ѱ_in_rs= lambda x: np.log(      2*np.cosh(x)       )
        ѱ_in_rs_int= lambda t: np.log(     integrate.quad(lambda z: (np.exp(-z*z/2)/np.sqrt(2*pi))*np.exp(m_*ѱ_in_rs(np.sqrt(q1_hat-qo_hat)*z+np.sqrt(qo_hat)*t)), -5, 5)[0]      )
        return    integrate.quad(lambda t: (np.exp(-t*t/2)/np.sqrt(2*pi))*ѱ_in_rs_int(t), -20, 20)[0]      /m_
        
    G_in=Ф_in(q1_hat,qo_hat,m_)
    

    return ((m_-1)/2)*q1*q1_hat-m_*qo*qo_hat+q1_hat/2-α*G_out-G_in



def Free_energy_1RSB_zero_field(q1,q1_hat,m_,κ,α):
    
    def Ф_out(Q,q1,m,κ):
        
        ѱ_out_rs= lambda x: np.log(      0.5*special.erf( (κ-x)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+x)/np.sqrt(2*(1-q1)) )        )
        func_out= lambda z: np.exp(-z*z/2+m_*ѱ_out_rs(np.sqrt(q1)*z))/np.sqrt(2*pi)
        
        cut=100
        dcut=10**(-0)
        N_cut=5000
        for j in range(N_cut):
            cut+=-dcut
            if abs(func_out(cut))*10**15>1:
                cut+=+dcut
                break
            
        return np.log(     integrate.quad(func_out, -cut, cut)[0]      )/m_
    
    G_out=Ф_out(1,q1,m_,κ)
    
    
    def Ф_in(q1_hat,m_):
        
        zo=m_*np.sqrt(q1_hat)
        
        ѱ_in_rs=  lambda z: np.log(    1+np.exp(-2*np.sqrt(q1_hat)*abs(z))      )+ np.sqrt(q1_hat)*abs(z)
        
        func_in= lambda z: (np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))/np.sqrt(2*pi))

        cut_in=100
        dcut=10**(-0)
        N_cut=5000
        for j in range(N_cut):
            cut_in+=-dcut
            if abs(func_in(cut_in))*10**15>1:
                cut_in+=+dcut
                break
    
        return (  np.log(  integrate.quad(func_in, -cut_in, cut_in)[0]  )  -zo*zo/2+m_*ѱ_in_rs(zo)  )/m_
        
    G_in=Ф_in(q1_hat,m_)
    
    
    return  ((m_-1)/2)*q1*q1_hat+q1_hat/2-α*G_out-G_in
    


def Complexity_1RSB_saddle_zero_field(q1,q1_hat,m_,κ,α):
    
    exp_ѱ_out_rs=  lambda z:       (   0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )     )
    ѱ_out_rs=  lambda z:  np.log(      0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )        ) if exp_ѱ_out_rs(z)>10**(-80) else -80
        
    func_out_2=lambda z: (np.exp(-z*z/2)/np.sqrt(2*pi))             *np.exp(m_*ѱ_out_rs(z))
    func_out_3=lambda z: (np.exp(-z*z/2)/np.sqrt(2*pi))* ѱ_out_rs(z)*np.exp(m_*ѱ_out_rs(z))
    
    
    zo=m_*np.sqrt(q1_hat)

    ѱ_in_rs=  lambda z: np.log(    1+np.exp(-2*np.sqrt(q1_hat)*abs(z))      )+ np.sqrt(q1_hat)*abs(z)
    
    func_in_2=lambda z: (np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))/np.sqrt(2*pi))
    func_in_3=lambda z: (np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))/np.sqrt(2*pi))* ѱ_in_rs(z) 
    

    cut=100
    dcut=10**(-0)
    N_cut=5000
    for j in range(N_cut):
        cut+=-dcut
        if abs(func_out_2(cut))*10**18>1:
            cut+=+dcut
            break
        
    cut_in=100
    dcut=10**(-0)
    N_cut=5000
    for j in range(N_cut):
        cut_in+=-dcut
        if abs(func_in_2(cut_in))*10**18>1:
            cut_in+=+dcut
            break
        
        

        
    denom_out=integrate.quad(func_out_2, -cut, cut)[0]
    denom_in =integrate.quad(func_in_2,  -cut_in, cut_in)[0]
    num_out_new=integrate.quad(func_out_3, -cut, cut)[0]
    num_in_new =integrate.quad(func_in_3 , -cut_in, cut_in)[0]
    
    eq_m=q1*q1_hat/2
    eq_m+=(α/m_**2)*np.log(denom_out)-(α/m_)*num_out_new/denom_out
    eq_m+=(1/m_**2)*(np.log(denom_in)-zo*zo/2+m_*ѱ_in_rs(zo))  -(1/m_)*num_in_new/denom_in  
        
    
    if print_test==3:
        N_test=500
        x_test=np.zeros(N_test)         
        y_test=np.zeros(N_test)  
        y_test2=np.zeros(N_test)  

        for j in range(N_test):
            x=-cut+2*j*abs(cut)/N_test
            x_test[j]=x
            y_test[j]=func_out_3(x)
            y_test2[j]=func_out_2(x)
                
        plt.plot(x_test,y_test)
        plt.plot(x_test,y_test2)
        plt.xlabel("out  "+str(np.log(denom_out))+"  "+str(num_out_new/denom_out)+"  "+str(eq_m))
        plt.ylabel(str(q1)+"  "+str((m_**2)*eq_m))
        plt.legend()
        plt.show()
    
    if print_test==3:
            N_test=500
            x_test=np.zeros(N_test)         
            y_test=np.zeros(N_test)  
            y_test2=np.zeros(N_test)  

            for j in range(N_test):
                x=-cut_in+2*j*abs(cut_in)/N_test

                x_test[j]=x
                y_test[j]=func_in_3(x)
                y_test2[j]=func_in_2(x)
                
            plt.plot(x_test,y_test)
            #plt.plot(x_test,y_test2)
            plt.xlabel("in  "+str(num_in_new ))
            plt.show()
        


    Σ=(m_**2)*eq_m
    return Σ
    
    



def Free_energy_1RSB_saddle_zero_field_iteration(q1,q1_hat,m_,κ,α):
    
    exp_ѱ_out_rs=  lambda z:       (   0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )     )
    ѱ_out_rs=      lambda z: np.log(   0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )     )
    dѱ_out_rs=     lambda z: (  (np.exp(-((κ-np.sqrt(q1)*z)**2)/(2*(1-q1)))  /  np.sqrt(pi))  *   (   -z/(2*np.sqrt(2*q1*(1-q1)))  +  (κ-np.sqrt(q1)*z)/(2*(1-q1))**(3/2)    )  +\
                                (np.exp(-((κ+np.sqrt(q1)*z)**2)/(2*(1-q1)))  /  np.sqrt(pi))  *   (   +z/(2*np.sqrt(2*q1*(1-q1)))  +  (κ+np.sqrt(q1)*z)/(2*(1-q1))**(3/2)    )   \
                                )/(exp_ѱ_out_rs(z)+10**(-15))
    func_out_1= lambda z: np.exp(-z*z/2)*dѱ_out_rs(z) *np.exp(m_*ѱ_out_rs(z))
    func_out_2= lambda z: np.exp(-z*z/2)              *np.exp(m_*ѱ_out_rs(z))


    zo=m_*np.sqrt(q1_hat)
    
    ѱ_in_rs=  lambda z: np.log(    1+np.exp(-2*np.sqrt(q1_hat)*abs(z))      )+ np.sqrt(q1_hat)*abs(z)

    
    dѱ_in_rs= lambda z: (z/(2*np.sqrt(q1_hat)))*np.tanh(np.sqrt(q1_hat)*z)   
    func_in_1=lambda z: np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))*dѱ_in_rs(z)
    func_in_2=lambda z: np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))
    
    if test==1:
        N_test=500
        x_list=np.zeros(N_test)
        y_list=np.zeros(N_test)
        xmin=-20
        for k in range(N_test):
            x_list[k]=xmin+2*abs(xmin)*k/N_test
            y_list[k]= func_in_2(x_list[k])

        plt.plot(x_list,y_list)
        plt.xlabel(t)
        plt.show()
            

    ########## saddle over q1 ##########
    ####################################
    cut=100
    dcut=10**(-0)
    N_cut=500000
    for j in range(N_cut):
        cut+=-dcut
        if abs(func_out_2(cut))*10**20>1:
            cut+=+dcut
            break
            
    
    num_out  =integrate.quad(func_out_1, -cut, cut)[0]
    denom_out=integrate.quad(func_out_2, -cut, cut)[0]
    
    
    q1_hat_out=-(2*α/(1-m_))*num_out/denom_out
    




    ########## saddle over q1_hat ##########
    ########################################
    cut_in=100
    dcut=10**(-0)
    N_cut=5000
    for j in range(N_cut):
        cut_in+=-dcut
        if abs(func_in_1(cut_in))*10**15>1:
            cut_in+=+dcut
            break
    
    
    num_in  =integrate.quad(func_in_1, -cut_in, cut_in)[0]
    denom_in=integrate.quad(func_in_2, -cut_in, cut_in)[0]
    
    q1_out=(2/(1-m_))*(1/2-num_in/denom_in)

    print("m:",m_)
    print("t,η:",t,η)
    print("q1,q1_hat",q1_out,q1_hat_out)
    print(num_out,denom_out)
    print(num_in,denom_in)

    print('')
    return [q1_out,q1_hat_out] 
  
def Free_energy_1RSB_saddle_zero_field(q1,q1_hat,m_,κ,α,m_saddle):
    
    exp_ѱ_out_rs=  lambda z:       (   0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )     )
    ѱ_out_rs=      lambda z: np.log(   0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )     )
    dѱ_out_rs=     lambda z: (  (np.exp(-((κ-np.sqrt(q1)*z)**2)/(2*(1-q1)))  /  np.sqrt(pi))  *   (   -z/(2*np.sqrt(2*q1*(1-q1)))  +  (κ-np.sqrt(q1)*z)/(2*(1-q1))**(3/2)    )  +\
                                (np.exp(-((κ+np.sqrt(q1)*z)**2)/(2*(1-q1)))  /  np.sqrt(pi))  *   (   +z/(2*np.sqrt(2*q1*(1-q1)))  +  (κ+np.sqrt(q1)*z)/(2*(1-q1))**(3/2)    )   \
                                )/(exp_ѱ_out_rs(z)+10**(-15))
    func_out_1= lambda z: np.exp(-z*z/2)*dѱ_out_rs(z) *np.exp(m_*ѱ_out_rs(z))
    func_out_13=lambda z: np.exp(-z*z/2)* ѱ_out_rs(z) *np.exp(m_*ѱ_out_rs(z))
    func_out_2= lambda z: np.exp(-z*z/2)              *np.exp(m_*ѱ_out_rs(z))

    
    
    zo=m_*np.sqrt(q1_hat)
    
    ѱ_in_rs=  lambda z: np.log(    1+np.exp(-2*np.sqrt(q1_hat)*abs(z))      )+ np.sqrt(q1_hat)*abs(z)

    
    dѱ_in_rs= lambda z: (z/(2*np.sqrt(q1_hat)))*np.tanh(np.sqrt(q1_hat)*z)   
    func_in_1=lambda z: np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))*dѱ_in_rs(z)
    func_in_2=lambda z: np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))
    func_in_3=lambda z: (np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))/np.sqrt(2*pi))* ѱ_in_rs(z) 
    
    
    
    ########## saddle over q1 ##########
    ####################################
    cut=100
    dcut=10**(-0)
    N_cut=500000
    for j in range(N_cut):
        cut+=-dcut
        if abs(func_out_2(cut))*10**20>1:
            cut+=+dcut
            break
   
    if print_test==0:
        N_test=500
        x_test=np.zeros(N_test)         
        y_test=np.zeros(N_test)  
        y_test2=np.zeros(N_test)  

        for j in range(N_test):
            x=-cut+2*j*abs(cut)/N_test
            x_test[j]=x
            y_test[j]=func_out_1(x)
            y_test2[j]=func_out_2(x)
        
        plt.plot(x_test,y_test)
        plt.plot(x_test,y_test2)
        plt.xlabel("out"+str(cut))
        plt.show()

    
    num_out  =integrate.quad(func_out_1, -cut, cut)[0]
    denom_out=integrate.quad(func_out_2, -cut, cut)[0]
    
    eq_q1=(m_-1)*q1_hat/2-α*num_out/(denom_out+10**(-25))
    




    ########## saddle over q1_hat ##########
    ########################################
    cut_in=100
    dcut=10**(-0)
    N_cut=5000
    for j in range(N_cut):
        cut_in+=-dcut
        if abs(func_in_1(cut_in))*10**15>1:
            cut_in+=+dcut
            break
    
    
    num_in  =integrate.quad(func_in_1, -cut_in, cut_in)[0]
    denom_in=integrate.quad(func_in_2, -cut_in, cut_in)[0]
    
    eq_q1_hat=((m_-1)/2)*q1+1/2-num_in/(denom_in+10**(-25))
    
    if print_test==0:
        N_test=500
        x_test=np.zeros(N_test)         
        y_test=np.zeros(N_test)  
        y_test2=np.zeros(N_test)  

        for j in range(N_test):
            x=-cut_in+2*j*abs(cut_in)/N_test

            x_test[j]=x
            y_test[j]=func_in_1(x)
            y_test2[j]=func_in_2(x)
        
        plt.plot(x_test,y_test)
        plt.plot(x_test,y_test2)
        plt.xlabel("in"+str(cut_in))
        plt.show()
    
    
    
    ########## saddle over m ##########
    ###################################
    if m_saddle==1:
        Σ_int=Complexity_1RSB_saddle_zero_field(q1,q1_hat,m_,κ,α)
        print("Σ:",Σ_int)
        Φ_int=Free_energy_1RSB_zero_field(q1,q1_hat,m_,κ,α)
        s_=-Φ_int-Σ_int/m_
        eq_m=s-s_
        
        return [eq_q1,eq_q1_hat,eq_m]    
    
    else: 
        return [eq_q1,eq_q1_hat] 
    
def Find_1RSB_saddle(sol):
    
    
    if m_saddle!=0:
        q1=min(1-eps,abs(sol[0]))
        q1_hat=abs(sol[1])
        
        print('κ:',κ)
        print("q1,q1_hat:",q1,q1_hat)
        print("m:",m)
        out=Free_energy_1RSB_saddle_zero_field(q1,q1_hat,m,κ,α,m_saddle)
        print("out:",out[0],out[1])
        print("")
        
        if solver==0:
            return [out[0]*10**3,out[1]*10**4]
        else:
            return abs(out[0]**2)*10**12+abs(out[1]**2)*10**18
        
        
    if m_saddle==1:
        q1=min(1-eps,abs(sol[0]))
        q1_hat=abs(sol[1])
        m_=abs(sol[2])
        
    
        
        if solver==0:
            if math.isnan(q1)!=True and math.isnan(q1_hat)!=True and math.isnan(m_)!=True and abs(m_-1.)>10**(-4):
                
                print('κ:',κ)
                print("q1,q1_hat,m:",q1,q1_hat,m_)
                print('s (entropy):',s)
                out=Free_energy_1RSB_saddle_zero_field(q1,q1_hat,m_,κ,α,m_saddle)
                print("out:",out[0],out[1],out[2])   
                print("")
                
                return [out[0]*10**3,out[1]*10**3,out[2]*10**3]
            else:
                print('κ:',κ)
                print("q1,q1_hat,m:",q1,q1_hat,m_)
                print('s (entropy):',s)
                print("out:",'nan')   
                print("")
                
                return [10**12,10**12,10**12]
        
        else:
            if math.isnan(q1)!=True and math.isnan(q1_hat)!=True and math.isnan(m_)!=True and abs(m_-1.)>10**(-4):
                print('κ:',κ)
                print("q1,q1_hat,m:",q1,q1_hat,m_)
                print('s (entropy):',s)
                out=Free_energy_1RSB_saddle_zero_field(q1,q1_hat,m_,κ,α,m_saddle)
                print("out:",out[0],out[1],out[2])   
                print("")
                return abs(out[0])*10**2.5+abs(out[1])*10**4+abs(out[2])*10**5
            else:
                print('κ:',κ)
                print("q1,q1_hat,m:",q1,q1_hat,m_)
                print('s (entropy):',s)
                out=Free_energy_1RSB_saddle_zero_field(q1,q1_hat,m_,κ,α,m_saddle)
                print("out:",out[0],out[1],out[2])   
                print("")
                return 10**29





def Free_energy_1RSB_saddle_zero_field2(q1,q1_hat,m_,κ,α):
            
    zo=m_*np.sqrt(q1_hat)
    
    ѱ_in_rs=  lambda z: np.log(    1+np.exp(-2*np.sqrt(q1_hat)*abs(z))      )+ np.sqrt(q1_hat)*abs(z)
        
    ѱ_in_rs=  lambda z: np.log(      2*np.cosh(np.sqrt(q1_hat)*z)       )  
    dѱ_in_rs= lambda z: (z/(2*np.sqrt(q1_hat)))*np.tanh(np.sqrt(q1_hat)*z)  
    func_in_1=lambda z: np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))*dѱ_in_rs(z)
    func_in_2=lambda z: np.exp(-z*z/2+zo*zo/2+m_*ѱ_in_rs(z)-m_*ѱ_in_rs(zo))
    


    ########## saddle over q1_hat ##########
    ########################################
    cut_in=100
    dcut=10**(-0)
    N_cut=5000
    for j in range(N_cut):
        cut_in+=-dcut
        if abs(func_in_1(cut_in))*10**15>1:
            cut_in+=+dcut
            break
        
    if print_test==3:
        N_test=500
        x_test=np.zeros(N_test)         
        y_test=np.zeros(N_test)  
        y_test2=np.zeros(N_test)  

        for j in range(N_test):
            x=-cut_in+2*j*abs(cut_in)/N_test

            x_test[j]=x
            y_test[j]=func_in_1(x)
            y_test2[j]=func_in_2(x)
            
        plt.plot(x_test,y_test)
        plt.plot(x_test,y_test2)
        plt.xlabel("in"+str(cut_in))
        plt.show()
        
    num_in  =integrate.quad(func_in_1, -cut_in, cut_in)[0]
    denom_in=integrate.quad(func_in_2, -cut_in, cut_in)[0]
    
    
    return -(2/(m_-1))*(+1/2-num_in/denom_in)
        
def Free_energy_1RSB_saddle_zero_field2_(q1,q1_hat,m_,κ,α):
    
    exp_ѱ_out_rs=  lambda z:       (   0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )     )
    ѱ_out_rs=      lambda z: np.log(   0.5*special.erf( (κ-np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )    +     0.5*special.erf( (κ+np.sqrt(q1)*z)/np.sqrt(2*(1-q1)) )     )
    dѱ_out_rs=     lambda z: (  (np.exp(-((κ-np.sqrt(q1)*z)**2)/(2*(1-q1)))  /  np.sqrt(pi))  *   (   -z/(2*np.sqrt(2*q1*(1-q1)))  +  (κ-np.sqrt(q1)*z)/(2*(1-q1))**(3/2)    )  +\
                                (np.exp(-((κ+np.sqrt(q1)*z)**2)/(2*(1-q1)))  /  np.sqrt(pi))  *   (   +z/(2*np.sqrt(2*q1*(1-q1)))  +  (κ+np.sqrt(q1)*z)/(2*(1-q1))**(3/2)    )   \
                                )/(exp_ѱ_out_rs(z)+10**(-15))

    func_out_1= lambda z: np.exp(-z*z/2)*dѱ_out_rs(z) *np.exp(m_*ѱ_out_rs(z))
    func_out_2= lambda z: np.exp(-z*z/2)              *np.exp(m_*ѱ_out_rs(z))



    
    ########## saddle over q1 ##########
    ####################################
    cut=10
    dcut=10**(-3)
    N_cut=500000
    for j in range(N_cut):
        cut+=-dcut
        if abs(func_out_2(cut))*10**50>1:
            cut+=+dcut
            break


    num_out  =integrate.quad(func_out_1, -cut, cut)[0]
    denom_out=integrate.quad(func_out_2, -cut, cut)[0]
    
    eps_=1*10**(-25)
    eq_q1=(m_-1)*q1_hat/2-α*num_out/(denom_out+eps_)
    
    if print_test==3:
        N_test=500
        x_test=np.zeros(N_test)         
        y_test=np.zeros(N_test)  
        y_test2=np.zeros(N_test)  

        for j in range(N_test):
            x=-cut+2*j*abs(cut)/N_test
            x_test[j]=x
            y_test[j]=func_out_1(x)
            y_test2[j]=func_out_2(x)*4000
        zo=(  κ-np.sqrt(2*np.log(m_)*(1-q1))  )/np.sqrt(q1)
        plt.plot([-κ/np.sqrt(q1),-κ/np.sqrt(q1),κ/np.sqrt(q1),κ/np.sqrt(q1)],[0,1,1,0],label=str(m_))
        plt.plot([-zo,-zo,zo,zo],[0,1,1,0],label=str(num_out))
        plt.plot(x_test,y_test/max(y_test[:]),label=str(num_out))
        plt.plot(x_test,y_test2/max(y_test2[:]),label=str(denom_out))
        plt.xlabel("out  "+str(q1_hat)+"  "+str(eq_q1)+"  "+str(q1))
        #plt.xlim([-cut,-0.5*cut])
        plt.legend()
        plt.show()
            
    
    
    return eq_q1 

def Find_1RSB_saddle2(q1_hat):
    
        
        
        #print("q1,m:",q1,m)
        #print("q1_hat:",q1_hat)
        out=Free_energy_1RSB_saddle_zero_field2_(q1,q1_hat,m,κ,α)
        #print("out:",out[0])
        #print("")

        return abs(out[0]**2)*10**8
        





###### Parameters relative to α ######
######################################
α=0.01
κ_SAT=κ_SAT_finding(α)[0]
print(κ_SAT)


###### Parameters relative to κ ######
######################################
κ=0.05


###### Parameters for saddles points ######
###########################################
way=-1
m_saddle=0
solver=3
eps=1*10**(-8)
print_test=3


if m_saddle==0:
    
    mo=200
    dm=0.05
    Nm=1
    m_cut=Nm-1
    
    sol=[0.997,62]
    
    m_list=np.zeros(Nm)
    q1_list=np.zeros(Nm)
    q1_hat_list=np.zeros(Nm)
    Σ_list=np.zeros(Nm)
    dΣ_list=np.zeros(Nm)
    Φ_list=np.zeros(Nm)
    Φ_rs_list=np.zeros(Nm)
    s_list=np.zeros(Nm)


if m_saddle==-1:
    mo=0.1
    dm=0.04
    Nm=1000
    m_cut=Nm-1
    
    sol= [0.976,20]

if m_saddle==-2:
    mo=40
    dm=0.1
    Nm=400
    m_cut=Nm-1
    
    sol= [0.971 , 0.064]  
    sol_=[0.9998, 0.121]
    
if m_saddle==-3:
    mo=40
    dm=0.1
    Nm=400
    m_cut=Nm-1
    
    sol= [0.825, 0.043 ]  
    sol_=[0.919, 0.0505]


marker_list=['o','x','+','1','s','*','p','d','<','4', \
                'o','x','+','1','s','*','p','d','<','4', \
                'o','x','+','1','s','*','p','d','<','4', \
                'o','x','+','1','s','*','p','d','<','4', \
                'o','x','+','1','s','*','p','d','<','4', \
                'o','x','+','1','s','*','p','d','<','4']
   
   
    
    

# fsolve
if solver==0 and m_saddle==0:
    if way==+1:
        file=open('test4.txt','w')
        file.write("x    q1    q1_hat    free_en_1RSB    free_en_RS    complexity    entropy    error")
        file.write("\n")
    if way==-1:
        file=open('test3.txt','w')  
        
    for k in range(Nm):
        if way==+1:
            m=mo+k*dm
        if way==-1:
            m=mo-k*dm
        
        sol=optimize.fsolve(Find_1RSB_saddle,sol)
        if abs(Find_1RSB_saddle(sol)[0])+abs(Find_1RSB_saddle(sol)[1])>0.1:
            solver=1
            sol=optimize.fmin(Find_1RSB_saddle,sol)      
            solver=0
        if abs(Find_1RSB_saddle(sol)[0])+abs(Find_1RSB_saddle(sol)[1])>0.1:
            m_cut=k
            break
        
        
        q1_list[k]=min(1-eps,abs(sol[0]))
        q1_hat_list[k]=abs(sol[1])
        m_list[k]=m
            
        Σ_list[k]=Complexity_1RSB_saddle_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
        Φ_list[k]=Free_energy_1RSB_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
        Φ_rs_list[k]=Free_energy_RS(q1_list[k],q1_hat_list[k],κ,α)
        
        s_list[k]=-Φ_list[k]-Σ_list[k]/m_list[k]
        out=Free_energy_1RSB_saddle_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α,m_saddle)
        if k>0:
            dΣ_list[k]=-(Σ_list[k]-Σ_list[k-1])/(s_list[k]-s_list[k-1])
            
        file.write(str(m_list[k]))
        file.write("    ")
        file.write(str(q1_list[k]))
        file.write("    ")
        file.write(str(q1_hat_list[k]))
        file.write("    ")
        file.write(str(Φ_list[k]))
        file.write("    ")
        file.write("    ")
        file.write(str(Σ_list[k]))
        file.write("    ")
        file.write(str(s_list[k]))
        file.write("    ")
        file.write(str(abs(out[0])+abs(out[1])))
        file.write("\n")
   
# fmin
if solver==1 and m_saddle==0:
    if way==+1:
        file=open('test4.txt','w')
    if way==-1:
        file=open('test3.txt','w')  
        
    for k in range(Nm):
        if way==+1:
            m=mo+k*dm
        if way==-1:
            m=mo-k*dm#np.exp((Nm-k)*np.log(mo)/Nm)
        sol=optimize.fmin(Find_1RSB_saddle,sol)
            
            
        q1_list[k]=min(1-eps,abs(sol[0]))
        q1_hat_list[k]=abs(sol[1])
        m_list[k]=m
        
        Σ_list[k]=Complexity_1RSB_saddle_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
        #Φ_list[k]=Free_energy_1RSB_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
        #Φ_rs_list[k]=Free_energy_RS(q1_list[k],q1_hat_list[k],κ,α)
        print(Free_energy_1RSB_zero_field(0,0,m_list[k],κ,α))
        print(Free_energy_RS(0,0,κ,α))
        print(α*np.log(special.erf(κ/np.sqrt(2))),np.log(2))
        s_list[k]=-Φ_list[k]-Σ_list[k]/m_list[k]
        out=Free_energy_1RSB_saddle_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α,m_saddle)
        
        if k>0:
            dΣ_list[k]=(Σ_list[k]-Σ_list[k-1])/(s_list[k]-s_list[k-1])
    
    
        file.write(str(m_list[k]))
        file.write("    ")
        file.write(str(q1_list[k]))
        file.write("    ")
        file.write(str(q1_hat_list[k]))
        file.write("    ")
        file.write(str(Φ_list[k]))
        file.write("    ")
        file.write(str(Σ_list[k]))
        file.write("    ")
        file.write(str(s_list[k]))
        file.write("    ")
        file.write(str(abs(out[0])+abs(out[1])))
        file.write("\n")
          

#reading
if solver==2 and m_saddle==0: 
    
    if way==+1 or way==0:
        file=open('test4.txt','r')
        with open(r"test4.txt", 'r') as fp:
            lines = sum(1 for line in fp)
            
        m_list=np.zeros(lines)
        q1_list=np.zeros(lines)
        q1_hat_list=np.zeros(lines)
        Σ_list=np.zeros(lines)
        dΣ_list=np.zeros(lines)
        Φ_list=np.zeros(lines)
        Φ_rs_list=np.zeros(lines)
        s_list=np.zeros(lines)
        m_cut=lines
            
        for k in range(lines):
                
            line=file.readline()
            line_=line.split()
                
            m_list[k]=float(line_[0])
            q1_list[k]=float(line_[1])
            q1_hat_list[k]=float(line_[2])
            Φ_list[k]=float(line_[3])
            Σ_list[k]=float(line_[4])
            s_list[k]=float(line_[5])

            
    file.close()        
    if way==-1 or way==0:
        file=open('test3.txt','r') 
        with open(r"test3.txt", 'r') as fp:
            lines = sum(1 for line in fp)
            
        m_list2=np.zeros(lines)
        q1_list2=np.zeros(lines)
        q1_hat_list2=np.zeros(lines)
        Σ_list2=np.zeros(lines)
        dΣ_list2=np.zeros(lines)
        Φ_list2=np.zeros(lines)
        Φ_rs_list2=np.zeros(lines)
        s_list2=np.zeros(lines)
        m_cut2=lines-1
        
        for k in range(lines):
        
            line=file.readline()
            line_=line.split()

        
            m_list2[k]=float(line_[0])
            q1_list2[k]=float(line_[1])
            q1_hat_list2[k]=float(line_[2])
            Φ_list2[k]=float(line_[3])
            Σ_list2[k]=float(line_[5])
            s_list2[k]=float(line_[6])

    file.close()         
    
    if way==0:
        file=open('Entropic evolution.txt','r') 
        file.readline()
        with open(r"Entropic evolution.txt", 'r') as fp:
            lines = sum(1 for line in fp)
            
        q1_list3=np.zeros(lines)
        Σ_list3=np.zeros(lines)
        s_list3=np.zeros(lines)
        m_cut3=lines-2
        
        for k in range(lines-1):
        
            line=file.readline()
            line_=line.split()
            print(line_)

    
            q1_list3[k]=float(line_[3])
            Σ_list3[k]=float(line_[5])
            s_list3[k]=float(line_[6])

    file.close() 
        
#q vs g(f(q))
if solver==3 and m_saddle==0:
    Nq=1
    qmax=0.999
    qmin=0.998
    
    q1_list=np.zeros(Nq)
    q1_hat_list=np.zeros(Nq)
    q1_out_list=np.zeros(Nq)
    m_list=np.zeros(Nq)
    Σ_list=np.zeros(Nq)
    dΣ_list=np.zeros(Nq)
    Φ_list=np.zeros(Nq)
    Φ_rs_list=np.zeros(Nq)
    s_list=np.zeros(Nq)
    
    index_stab=np.zeros(16)
    eps_q=3*10**(-16)
    index=0

    m=mo
    for k in range(Nq):
            
            q1=min(1-eps,qmin+(qmax-qmin)*(k+1)/Nq)
            if k==0:
                q1_hat=0.1
            
            q1_hat=optimize.fsolve(Find_1RSB_saddle2,q1_hat)[0]
            q1_list[k]=q1
            q1_hat_list[k]=q1_hat
            q1_out_list[k]=Free_energy_1RSB_saddle_zero_field2(q1,q1_hat,m,κ,α)
            m_list[k]=m
            
            Σ_list[k]=Complexity_1RSB_saddle_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
            Φ_list[k]=Free_energy_1RSB_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
            
            s_list[k]=-Φ_list[k]-Σ_list[k]/m_list[k]
            print(k,q1_hat)
            print(q1,q1_out_list[k]-q1)
            if abs(q1-q1_out_list[k])<eps_q:
                index_stab[index]=k
                index+=1
                
                
        
        
    plt.plot(q1_list[:],q1_out_list[:],label="x="+str(round(m,4))+"  κ="+str(round(κ,4)))
    
   
    plt.xlabel("q1")
    plt.ylabel("q1")
    print(q1_list[:])
    plt.plot(q1_list,q1_list,c='black',linestyle='--')
    plt.legend()
    plt.ylim([qmin,qmax])
    plt.show()
    
    plt.plot(q1_list,q1_hat_list,label="x="+str(round(m,4)))
    plt.xlabel("q1")
    plt.ylabel("q1_hat")
    plt.show()
    
    #print("s",s_list)
    #print("Σ",Σ_list)
    #print('Φ',Φ_list)
    plt.plot(q1_list[:],s_list[:],label="x="+str(round(m,4)))
    plt.xlabel("q1")
    plt.ylabel("s")
    plt.show()
    
    plt.plot(q1_list[:],Σ_list[:],label="x="+str(round(m,4)))
    plt.xlabel("q1")
    plt.ylabel("Σ")
    #y_min=min(Σ_list[i][:])
    #y_max=max(Σ_list[i][:])
    #plt.ylim([y_min,y_max])
    plt.show()
    
    plt.plot(q1_list[:],(-Φ_list[:]),label="x="+str(round(m,4)))
    plt.xlabel("q1")
    plt.ylabel("-Φ")
    #y_min=min(-Φ_list[:]*m_list[:])
    #y_max=max(-Φ_list[:]*m_list[:])
    #plt.ylim([y_min,y_max])
    plt.show()  

#iteration
if solver==4 and m_saddle==0:
    Nt=150
    η=0.01
    
    q1=sol[0]
    q1_hat=sol[1]
    
    for k in range(Nm):
        if way==+1:
            m=mo+k*dm
        if way==-1:
            m=mo-k*dm
            
        for t in range(Nt):
            
            if t==0 or t==int(Nt/2):
                test=1
            else:
                test=0
            q1_new,q1_hat_new=Free_energy_1RSB_saddle_zero_field_iteration(q1,q1_hat,m,κ,α)
            q1    =(1-η)*q1    +η*q1_new
            q1_hat=(1-η)*q1_hat+η*q1_hat_new 
            
                    
        q1_list[k]=q1
        q1_hat_list[k]=q1_hat 
        m_list[k]=m
            
        Σ_list[k]=Complexity_1RSB_saddle_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
        Φ_list[k]=Free_energy_1RSB_zero_field(q1_list[k],q1_hat_list[k],m_list[k],κ,α)
        s_list[k]=-Φ_list[k]-Σ_list[k]/m_list[k]
    
# fixing the entropy
if m_saddle==1:
    Ns=300
    
    m_list=np.zeros((Nκ,Ns))
    q1_list=np.zeros((Nκ,Ns))
    q1_hat_list=np.zeros((Nκ,Ns))
    Σ_list=np.zeros((Nκ,Ns))
    dΣ_list=np.zeros((Nκ,Ns))
    Φ_list=np.zeros((Nκ,Ns))
    Φ_rs_list=np.zeros((Nκ,Ns))
    s_list=np.zeros((Nκ,Ns))    

    
    for i in range(Nκ):
        κ=κmin+(κmax-κmin)*i/Nκ
        
    
        file=open('saddle_branches(final_)(α='+str(α)+',κ='+str(κ)+').txt','w')
        file.write("x    q1    q1_hat    free_en_1RSB    complexity    entropy    error")
        file.write("\n")
    
        breaking_sol=0
        
        ratio_sol=1
    
    
        smax=0.028
        smin=0.0005

        
        for k in range(Ns):
        
            s=smax-(smax-smin)*k/Ns

            if i!=0:
                if  k!=0 :
                    sol[0]=q1_list[i-1][k]*(1-ratio_sol)    +sol[0]*ratio_sol
                    sol[1]=q1_hat_list[i-1][k]*(1-ratio_sol)+sol[1]*ratio_sol
                    sol[2]=m_list[i-1][k]*(1-ratio_sol)     +sol[2]*ratio_sol
                if k==0:
                    sol=[q1_list[i-1][k],q1_hat_list[i-1][k],m_list[i-1][k]]
                    


                
            if breaking_sol==0:
                
                if solver==0 and breaking_sol==0:
                    sol=optimize.fsolve(Find_1RSB_saddle,sol)
                if solver==1 and breaking_sol==0:
                    sol=optimize.fmin(Find_1RSB_saddle,sol)
                
                
                q1_list[i][k]=min(1-eps,abs(sol[0]))
                q1_hat_list[i][k]=abs(sol[1])
                m_list[i][k]=abs(sol[2])
                out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
                
                if  solver==0 and abs(out[0])+abs(out[1])+abs(out[2])>10**(-3):
                    solver=1
                    sol=optimize.fmin(Find_1RSB_saddle,sol)
                    solver=0
        
                q1_list[i][k]=min(1-eps,abs(sol[0]))
                q1_hat_list[i][k]=abs(sol[1])
                m_list[i][k]=abs(sol[2])
                out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
                
                if  abs(out[0])+abs(out[1])+abs(out[2])>0.05:
                    breaking_sol=1
                
            else:
                
                q1_list[i][k]=q1_list[i][k-1]
                q1_hat_list[i][k]=q1_hat_list[i][k-1]
                m_list[i][k]=m_list[i][k-1]
                
    
        
            Σ_list[i][k]=Complexity_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            Φ_list[i][k]=Free_energy_1RSB_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            s_list[i][k]=-Φ_list[i][k]-Σ_list[i][k]/m_list[i][k]
            out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
        
            file.write(str(m_list[i][k]))
            file.write("    ")
            file.write(str(q1_list[i][k]))
            file.write("    ")
            file.write(str(q1_hat_list[i][k]))
            file.write("    ")
            file.write(str(Φ_list[i][k]))
            file.write("    ")
            file.write("    ")
            file.write(str(Σ_list[i][k]))
            file.write("    ")
            file.write(str(s_list[i][k]))
            file.write("    ")
            file.write(str(abs(out[0])+abs(out[1])+abs(out[2])))
            file.write("\n")
     
        m_cut=Ns-1
        file.close()   
        
# Getting the lower branch
if m_saddle==-1:
    κmin=0.5
    κmax=0.6
    Nκ=5
    
    m_list=np.zeros((Nκ,Nm))
    q1_list=np.zeros((Nκ,Nm))
    q1_hat_list=np.zeros((Nκ,Nm))
    Σ_list=np.zeros((Nκ,Nm))
    dΣ_list=np.zeros((Nκ,Nm))
    Φ_list=np.zeros((Nκ,Nm))
    Φ_rs_list=np.zeros((Nκ,Nm))
    s_list=np.zeros((Nκ,Nm))    

    
    for i in range(Nκ):
        κ=κmin+(κmax-κmin)*i/Nκ
        
    
        file=open('saddle_branches(final3)(α='+str(α)+',κ='+str(κ)+').txt','w')
        file.write("x    q1    q1_hat    free_en_1RSB    complexity    entropy    error")
        file.write("\n")
    
        breaking_sol=0
        
        ratio_sol=0
    
        for k in range(Nm):
        
            m=mo+dm*k
            
            if i!=0 and k==0:
                sol=[q1_list[i-1][k],q1_hat_list[i-1][k]]
            

         
                
            if solver==0 and breaking_sol==0:
                sol=optimize.fsolve(Find_1RSB_saddle,sol)
            if solver==1 and breaking_sol==0:
                sol=optimize.fmin(Find_1RSB_saddle,sol)
                    
                    
            q1_list[i][k]=min(1-eps,abs(sol[0]))
            q1_hat_list[i][k]=abs(sol[1])
            m_list[i][k]=m
            out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
                
            if  solver==0 and abs(out[0])+abs(out[1])>10**(-3):
                solver=1
                sol=optimize.fmin(Find_1RSB_saddle,sol)
                solver=0
        
            q1_list[i][k]=min(1-eps,abs(sol[0]))
            q1_hat_list[i][k]=abs(sol[1])
            m_list[i][k]=m
            out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
                
        
        
                
    
        
            Σ_list[i][k]=Complexity_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            Φ_list[i][k]=Free_energy_1RSB_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            s_list[i][k]=-Φ_list[i][k]-Σ_list[i][k]/m_list[i][k]
            out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
        
            file.write(str(m_list[i][k]))
            file.write("    ")
            file.write(str(q1_list[i][k]))
            file.write("    ")
            file.write(str(q1_hat_list[i][k]))
            file.write("    ")
            file.write(str(Φ_list[i][k]))
            file.write("    ")
            file.write("    ")
            file.write(str(Σ_list[i][k]))
            file.write("    ")
            file.write(str(s_list[i][k]))
            file.write("    ")
            file.write(str(abs(out[0])+abs(out[1])))
            file.write("\n")
     
        m_cut=Nm-1
        file.close() 

# Getting the doublet at low κ
if m_saddle==-2:
    κmin=0.4
    κmax=0.5
    Nκ=20
    
    m_list=np.zeros((Nκ,Nm))
    q1_list=np.zeros((Nκ,Nm))
    q1_hat_list=np.zeros((Nκ,Nm))
    Σ_list=np.zeros((Nκ,Nm))
    dΣ_list=np.zeros((Nκ,Nm))
    Φ_list=np.zeros((Nκ,Nm))
    Φ_rs_list=np.zeros((Nκ,Nm))
    s_list=np.zeros((Nκ,Nm))  
    
    m_list2=np.zeros((Nκ,Nm))
    q1_list2=np.zeros((Nκ,Nm))
    q1_hat_list2=np.zeros((Nκ,Nm))
    Σ_list2=np.zeros((Nκ,Nm))
    dΣ_list2=np.zeros((Nκ,Nm))
    Φ_list2=np.zeros((Nκ,Nm))
    Φ_rs_list2=np.zeros((Nκ,Nm))
    s_list2=np.zeros((Nκ,Nm))   

    
    for i in range(Nκ):
        κ=κmin+(κmax-κmin)*i/Nκ
        
    
        file=open('saddle_branches(final1)(α='+str(α)+',κ='+str(κ)+').txt','w')
        file.write("x    q1    q1_hat    free_en_1RSB    complexity    entropy    error")
        file.write("\n")
        
        file2=open('saddle_branches(final2)(α='+str(α)+',κ='+str(κ)+').txt','w')
        file2.write("x    q1    q1_hat    free_en_1RSB    complexity    entropy    error")
        file2.write("\n")
    
        breaking_sol=0
        
        ratio_sol=0
    
        for k in range(Nm):
        
            m=mo-dm*k
            
            if i!=0 and k==0:
                sol =[q1_list[i-1][k] ,q1_hat_list[i-1][k]]
                sol_=[q1_list2[i-1][k],q1_hat_list2[i-1][k]]
            

            
            if breaking_sol==0:
                
                if solver==0 and breaking_sol==0:
                    sol=optimize.fsolve(Find_1RSB_saddle,sol)
                    sol_=optimize.fsolve(Find_1RSB_saddle,sol_)
                if solver==1 and breaking_sol==0:
                    sol=optimize.fmin(Find_1RSB_saddle,sol)
                    sol_=optimize.fmin(Find_1RSB_saddle,sol_)
                    
                    
                q1_list[i][k]=min(1-eps,abs(sol[0]))
                q1_hat_list[i][k]=abs(sol[1])
                m_list[i][k]=m
                
                q1_list2[i][k]=min(1-eps,abs(sol_[0]))
                q1_hat_list2[i][k]=abs(sol_[1])
                m_list2[i][k]=m
                
                out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k]  ,q1_hat_list[i][k] ,m_list[i][k] ,κ,α,m_saddle)
                out2=Free_energy_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α,m_saddle)
                
                if  solver==0 and abs(out[0])+abs(out[1])>10**(-3):
                    solver=1
                    sol=optimize.fmin(Find_1RSB_saddle,sol)
                    solver=0
                    
                if  solver==0 and abs(out2[0])+abs(out2[1])>10**(-3):
                    solver=1
                    sol_=optimize.fmin(Find_1RSB_saddle,sol_)
                    solver=0
        
                q1_list[i][k]=min(1-eps,abs(sol[0]))
                q1_hat_list[i][k]=abs(sol[1])
                m_list[i][k]=m
                q1_list2[i][k]=min(1-eps,abs(sol_[0]))
                q1_hat_list2[i][k]=abs(sol_[1])
                m_list2[i][k]=m
                
                out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
                out2=Free_energy_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α,m_saddle)
                
                if  abs(out[0])+abs(out[1])>0.05 or abs(out2[0])+abs(out2[1])>0.05 or abs(q1_list[i][k]-q1_list2[i][k])<10**(-4):
                    breaking_sol=1
                
            else:
                
                q1_list[i][k]=q1_list[i][k-1]
                q1_hat_list[i][k]=q1_hat_list[i][k-1]
                m_list[i][k]=m_list[i][k-1]
                
                q1_list2[i][k]=q1_list2[i][k-1]
                q1_hat_list2[i][k]=q1_hat_list2[i][k-1]
                m_list2[i][k]=m_list2[i][k-1]
                
    
        
            Σ_list[i][k]=Complexity_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            Φ_list[i][k]=Free_energy_1RSB_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            s_list[i][k]=-Φ_list[i][k]-Σ_list[i][k]/m_list[i][k]
            out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
        
            file.write(str(m_list[i][k]))
            file.write("    ")
            file.write(str(q1_list[i][k]))
            file.write("    ")
            file.write(str(q1_hat_list[i][k]))
            file.write("    ")
            file.write(str(Φ_list[i][k]))
            file.write("    ")
            file.write("    ")
            file.write(str(Σ_list[i][k]))
            file.write("    ")
            file.write(str(s_list[i][k]))
            file.write("    ")
            file.write(str(abs(out[0])+abs(out[1])))
            file.write("\n")
            
            Σ_list2[i][k]=Complexity_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α)
            Φ_list2[i][k]=Free_energy_1RSB_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α)
            s_list2[i][k]=-Φ_list2[i][k]-Σ_list2[i][k]/m_list2[i][k]
            out2=Free_energy_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α,m_saddle)
        
            file2.write(str(m_list2[i][k]))
            file2.write("    ")
            file2.write(str(q1_list2[i][k]))
            file2.write("    ")
            file2.write(str(q1_hat_list2[i][k]))
            file2.write("    ")
            file2.write(str(Φ_list2[i][k]))
            file2.write("    ")
            file2.write("    ")
            file2.write(str(Σ_list2[i][k]))
            file2.write("    ")
            file2.write(str(s_list2[i][k]))
            file2.write("    ")
            file2.write(str(abs(out2[0])+abs(out2[1])))
            file2.write("\n")
     
        m_cut=Nm-1
        file.close() 
        file2.close() 

# Getting the doublet at high κ
if m_saddle==-3:
    κmin=0.5
    κmax=0.4
    Nκ=20
    
    m_list=np.zeros((Nκ,Nm))
    q1_list=np.zeros((Nκ,Nm))
    q1_hat_list=np.zeros((Nκ,Nm))
    Σ_list=np.zeros((Nκ,Nm))
    dΣ_list=np.zeros((Nκ,Nm))
    Φ_list=np.zeros((Nκ,Nm))
    Φ_rs_list=np.zeros((Nκ,Nm))
    s_list=np.zeros((Nκ,Nm))  
    
    m_list2=np.zeros((Nκ,Nm))
    q1_list2=np.zeros((Nκ,Nm))
    q1_hat_list2=np.zeros((Nκ,Nm))
    Σ_list2=np.zeros((Nκ,Nm))
    dΣ_list2=np.zeros((Nκ,Nm))
    Φ_list2=np.zeros((Nκ,Nm))
    Φ_rs_list2=np.zeros((Nκ,Nm))
    s_list2=np.zeros((Nκ,Nm))   

    
    for i in range(Nκ):
        κ=κmin+(κmax-κmin)*i/Nκ
        
    
        file=open('saddle_branches(final11)(α='+str(α)+',κ='+str(κ)+').txt','w')
        file.write("x    q1    q1_hat    free_en_1RSB    complexity    entropy    error")
        file.write("\n")
        
        file2=open('saddle_branches(final22)(α='+str(α)+',κ='+str(κ)+').txt','w')
        file2.write("x    q1    q1_hat    free_en_1RSB    complexity    entropy    error")
        file2.write("\n")
    
        breaking_sol=0
        
        ratio_sol=0
    
        for k in range(Nm):
        
            m=mo-dm*k
            
            if i!=0 and k==0:
                sol =[q1_list[i-1][k] ,q1_hat_list[i-1][k]]
                sol_=[q1_list2[i-1][k],q1_hat_list2[i-1][k]]
            

            
            if breaking_sol==0:
                
                if solver==0 and breaking_sol==0:
                    sol=optimize.fsolve(Find_1RSB_saddle,sol)
                    sol_=optimize.fsolve(Find_1RSB_saddle,sol_)
                if solver==1 and breaking_sol==0:
                    sol=optimize.fmin(Find_1RSB_saddle,sol)
                    sol_=optimize.fmin(Find_1RSB_saddle,sol_)
                    
                    
                q1_list[i][k]=min(1-eps,abs(sol[0]))
                q1_hat_list[i][k]=abs(sol[1])
                m_list[i][k]=m
                
                q1_list2[i][k]=min(1-eps,abs(sol_[0]))
                q1_hat_list2[i][k]=abs(sol_[1])
                m_list2[i][k]=m
                
                out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k]  ,q1_hat_list[i][k] ,m_list[i][k] ,κ,α,m_saddle)
                out2=Free_energy_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α,m_saddle)
                
                if  solver==0 and abs(out[0])+abs(out[1])>10**(-3):
                    solver=1
                    sol=optimize.fmin(Find_1RSB_saddle,sol)
                    solver=0
                    
                if  solver==0 and abs(out2[0])+abs(out2[1])>10**(-3):
                    solver=1
                    sol_=optimize.fmin(Find_1RSB_saddle,sol_)
                    solver=0
        
                q1_list[i][k]=min(1-eps,abs(sol[0]))
                q1_hat_list[i][k]=abs(sol[1])
                m_list[i][k]=m
                q1_list2[i][k]=min(1-eps,abs(sol_[0]))
                q1_hat_list2[i][k]=abs(sol_[1])
                m_list2[i][k]=m
                
                out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
                out2=Free_energy_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α,m_saddle)
                
                if  abs(out[0])+abs(out[1])>0.05 or abs(out2[0])+abs(out2[1])>0.05 or abs(q1_list[i][k]-q1_list2[i][k])<10**(-4):
                    breaking_sol=1
                
            else:
                
                q1_list[i][k]=q1_list[i][k-1]
                q1_hat_list[i][k]=q1_hat_list[i][k-1]
                m_list[i][k]=m_list[i][k-1]
                
                q1_list2[i][k]=q1_list2[i][k-1]
                q1_hat_list2[i][k]=q1_hat_list2[i][k-1]
                m_list2[i][k]=m_list2[i][k-1]
                
    
        
            Σ_list[i][k]=Complexity_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            Φ_list[i][k]=Free_energy_1RSB_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α)
            s_list[i][k]=-Φ_list[i][k]-Σ_list[i][k]/m_list[i][k]
            out=Free_energy_1RSB_saddle_zero_field(q1_list[i][k],q1_hat_list[i][k],m_list[i][k],κ,α,m_saddle)
        
            file.write(str(m_list[i][k]))
            file.write("    ")
            file.write(str(q1_list[i][k]))
            file.write("    ")
            file.write(str(q1_hat_list[i][k]))
            file.write("    ")
            file.write(str(Φ_list[i][k]))
            file.write("    ")
            file.write("    ")
            file.write(str(Σ_list[i][k]))
            file.write("    ")
            file.write(str(s_list[i][k]))
            file.write("    ")
            file.write(str(abs(out[0])+abs(out[1])))
            file.write("\n")
            
            Σ_list2[i][k]=Complexity_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α)
            Φ_list2[i][k]=Free_energy_1RSB_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α)
            s_list2[i][k]=-Φ_list2[i][k]-Σ_list2[i][k]/m_list2[i][k]
            out2=Free_energy_1RSB_saddle_zero_field(q1_list2[i][k],q1_hat_list2[i][k],m_list2[i][k],κ,α,m_saddle)
        
            file2.write(str(m_list2[i][k]))
            file2.write("    ")
            file2.write(str(q1_list2[i][k]))
            file2.write("    ")
            file2.write(str(q1_hat_list2[i][k]))
            file2.write("    ")
            file2.write(str(Φ_list2[i][k]))
            file2.write("    ")
            file2.write("    ")
            file2.write(str(Σ_list2[i][k]))
            file2.write("    ")
            file2.write(str(s_list2[i][k]))
            file2.write("    ")
            file2.write(str(abs(out2[0])+abs(out2[1])))
            file2.write("\n")
     
        m_cut=Nm-1
        file.close() 
        file2.close() 










#Printing part
if solver!=3 and m_saddle!=1:    
    
    plt.scatter(s_list[0:m_cut],Σ_list[0:m_cut],c=m_list[0:m_cut],s=0.8,label='1RSB saddle') 
    if m_saddle==-2:
        plt.scatter(s_list2[0:m_cut],Σ_list2[0:m_cut],c=m_list2[0:m_cut],s=0.8,label='1RSB saddle') 
    
    plt.xlabel("s (entropy)")
    plt.ylabel("Σ (complexity)")
    plt.legend()
    plt.savefig('fig0.png')
    plt.show()

    plt.scatter(q1_list[0:m_cut],Σ_list[0:m_cut],c=m_list[0:m_cut],s=0.8,label='1RSB saddle') 
    
    plt.xlabel("q1 (overlap)")
    plt.ylabel("Σ (complexity)")
    plt.legend()
    plt.savefig('fig1.png')
    plt.show()   

    plt.scatter(m_list[0:m_cut],s_list[0:m_cut],c=m_list[0:m_cut],s=0.8) 
    if m_saddle==-2:
        plt.scatter(m_list2[0:m_cut],s_list2[0:m_cut],c=m_list2[0:m_cut],s=0.8,label='1RSB saddle') 
    plt.xlabel("x (Parisi parameter)")
    plt.ylabel("s (Entropy)")
    cbar = plt.colorbar()
    cbar.set_label('x (Parisi parameter)')
    plt.savefig('fig4.png')
    plt.show()    
    
    plt.scatter(m_list[0:m_cut],q1_hat_list[0:m_cut],c=m_list[0:m_cut],s=0.8) 
    plt.xlabel("x (Parisi parameter)")
    plt.ylabel("q1_hat (intra-cluster overlap)")
    cbar = plt.colorbar()
    cbar.set_label('x (Parisi parameter)')
    plt.savefig('fig6.png')
    plt.show()  


if m_saddle==1:
    for i in range(Nκ): 
        κ=κmin+(κmax-κmin)*i/Nκ
        plt.scatter(s_list[i][0:m_cut],Σ_list[i][0:m_cut],c=m_list[i][0:m_cut],s=1.2,marker=marker_list[i],label='κ='+str(round(κ,5))) 
    plt.xlabel("s (entropy)")
    plt.ylabel("Σ (complexity)")
    cbar = plt.colorbar()
    cbar.set_label('x (Parisi parameter)')
    plt.legend()
    plt.savefig('fig0.png')
    plt.show()

    for i in range(Nκ): 
        κ=κmin+(κmax-κmin)*i/Nκ
        plt.scatter(q1_list[i][0:m_cut],Σ_list[i][0:m_cut],c=m_list[i][0:m_cut],s=1.2,marker=marker_list[i],label='κ='+str(round(κ,5))) 
        
    plt.xlabel("q1 (overlap)")
    plt.ylabel("Σ (complexity)")
    cbar = plt.colorbar()
    cbar.set_label('x (Parisi parameter)')
    plt.legend()
    plt.savefig('fig1.png')
    plt.show() 

    for i in range(Nκ): 
        κ=κmin+(κmax-κmin)*i/Nκ
        plt.scatter(m_list[i][0:m_cut],s_list[i][0:m_cut],c=m_list[i][0:m_cut],s=1.2,marker=marker_list[i],label='κ='+str(round(κ,5)))
    plt.xlabel("x (Parisi parameter)")
    plt.ylabel("s (Entropy)")
    cbar = plt.colorbar()
    cbar.set_label('x (Parisi parameter)')
    plt.savefig('fig4.png')
    plt.show()    
    
    for i in range(Nκ): 
        κ=κmin+(κmax-κmin)*i/Nκ
        plt.scatter(m_list[i][0:m_cut],q1_list[i][0:m_cut],c=m_list[i][0:m_cut],s=1.2,marker=marker_list[i],label='κ='+str(round(κ,5)))  
    plt.xlabel("x (Parisi parameter)")
    plt.ylabel("q1 (intra-cluster overlap)")
    cbar = plt.colorbar()
    cbar.set_label('x (Parisi parameter)')
    plt.savefig('fig5.png')
    plt.show()   
    
    for i in range(Nκ): 
        κ=κmin+(κmax-κmin)*i/Nκ
        plt.scatter(m_list[i][0:m_cut],q1_hat_list[i][0:m_cut],c=m_list[i][0:m_cut],s=1.2,marker=marker_list[i],label='κ='+str(round(κ,5)))  
    plt.xlabel("x (Parisi parameter)")
    plt.ylabel("q1_hat (intra-cluster overlap)")
    cbar = plt.colorbar()
    cbar.set_label('x (Parisi parameter)')
    plt.savefig('fig6.png')
    plt.show()   




    
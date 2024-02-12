

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358

#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)


////////////////// Random numbers ///////////////////////

// Routine for random normal numbers (number 1)
/* variabili globali per il generatore random */
unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

unsigned randForInit(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000) {
    myrand = (myrand & 0x7fffffff) + 1;
  }
  return myrand;
}

void initRandom(void) {
  int i;
  
  ip = 128;    
  ip1 = ip - 24;    
  ip2 = ip - 55;    
  ip3 = ip - 61;
  
  for (i = ip3; i < ip; i++) {
    ira[i] = randForInit();
  }
}

float gaussRan(void) {
  static int iset = 0;
  static float gset;
  float fac, rsq, v1, v2;
  
  if (iset == 0) {
    do {
      v1 = 2.0 * FRANDOM - 1.0;
      v2 = 2.0 * FRANDOM - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  } else {
    iset = 0;
    return gset;
  }
}




// Routine for random normal numbers (number 2)
double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r+0.00000000000000000000000000001) / (r+0.00000000000000000000000000001));
    return u * c;
}






// Function generating a Gaussian random Matrix M*N with mean 0 and variance 1/N
void sampleNormal_matrix(double **J,int M,int N){
  int i,j;


  for (i=0;i<M;i++){
    for (j=0;j<N;j++){
      J[i][j]=gaussRan()/sqrt(N);
    }
  }
}

/////////////////////////////////////////////////////////













////////////////// Building the interaction matrix ///////////////////////

void setting_interaction(double **J,int M,int N,double alpha,double k_old){
  int i,j;

  printf("Generating a Gaussian random Matrix M*N with mean 0 and variance 1/N \n");
  printf("\n");
 
  for (i=0;i<M;i++)
    J[i]= (double *) malloc(N*sizeof(double));

  sampleNormal_matrix(J,M,N);

  printf("Computing the interraction of xo=[+1,...,+1] and substracting it from J \n");
  printf("\n");
  double * field_xo= (double *) malloc(M*sizeof(double));

  for (i=0;i<M;i++){
    field_xo[i]=0;

    for (j=0;j<N;j++){
      field_xo[i]+=J[i][j];
    }
    for (j=0;j<N;j++){
      J[i][j]-=field_xo[i]/N;
    }
  }

  printf("Adding the interraction of xo=[+1,...,+1] in J \n");
  printf("\n");
  double * field_xo_new= (double *) malloc(M*sizeof(double));


  for (i=0;i<M;i++){
    int accept = 0;
    while (accept == 0 ){
      field_xo_new[i]=sampleNormal();

      if (fabs(field_xo_new[i])<k_old){
        for (j=0;j<N;j++){
	      J[i][j]+=field_xo_new[i]/N;
        }
	      accept=1;
      }

    }
  }

}

/////////////////////////////////////////////////////////
















////////////////// Running the perceptron Quench ///////////////////////
void perceptron_quench(double *mag,double *time_rescaled,double **J,int M,int N,int index_try,double alpha,double k_old,double k_new){
  int i,j;
  int Total_time=1500*N;
  int delta_time_memory=N/10;

  int Number_of_spin_flipped=0;




  //// initialization de x////
  double * x= (double *) malloc(N*sizeof(double));
  for (i=0;i<N;i++){
    x[i]=1;
  }

  //// initialization des interactions////
  double * field= (double *) malloc(M*sizeof(double));
  for (i=0;i<M;i++){
    field[i]=0;
    for (j=0;j<N;j++){
      field[i]+=J[i][j];
    }
  }


  //// Dynamics////
  for (i=0;i<Total_time;i++){
    int i_rand=rand()%N;
    int accept=1;

    if (i%(Total_time/100)==0){
      printf("Time:");
      printf("%d %d",i,Total_time);
      printf("\n");
      printf("k_new:");
      printf("%f",k_new);
      printf("\n");
    }

    //// Checking the spin flip validity ////
    for(j=0;j<M;j++){
      double field_temp=field[j]-2*J[j][i_rand]*x[i_rand];
      if (fabs(field_temp)>k_new){
        accept=0;
        break;
      }
    }

    //// Spin flip ?////
    if (accept==1){
      Number_of_spin_flipped+=1;
      for(j=0;j<M;j++){
        field[j]-=2*J[j][i_rand]*x[i_rand];
      }
      x[i_rand]*=-1;
    }

    //// Memory of the magnetization ////
    if (i%delta_time_memory==0){

      double mag_temp=0;
      for(j=0;j<N;j++){
        mag_temp+=x[j];
      }
      mag[i/delta_time_memory]=mag_temp/N;
      time_rescaled[i/delta_time_memory]=Number_of_spin_flipped;

    }


  }
}

////////////////////////////////////////////////////////////////////////



















// MAIN CODE
int main(){
  myrand = 12;
  initRandom();

  int i,j,k;

  double alpha,k_old;
  alpha=0.5;
  k_old=0.3186393639643758;

  int N_try_size_realization=1;
  int N_try_disorder_realization=2;

  double * N_list=  (double *) malloc(N_try_size_realization*sizeof(double));
  /// N_list[0]=1000;
  /// N_list[1]=3000; 
  /// N_list[2]=7000; 
  /// N_list[3]=10000; 
  /// N_list[4]=20000; 
  N_list[0]=40000; 

  double * k_new_list=  (double *) malloc(N_try_size_realization*sizeof(double));
  /// k_new_list[0]=0.8037353515624999;
  /// k_new_list[1]=0.9070068359374999; 
  /// k_new_list[2]=0.9827392578125; 
  /// k_new_list[3]=1.0148681640625; 
  /// k_new_list[4]=1.0791259765624999; 
  k_new_list[0]=1.1548583984375; 


  // file pointer variable to store the value returned by
  // fopen
  FILE* fptr;
 
  // opening the file in read mode
  fptr = fopen("mag_fixed_m3.txt", "w");
  fprintf(fptr, "Try_index N kappa_old kappa_new time_rescaled mag");
  fprintf(fptr, "\n");



  for (i=0;i<N_try_size_realization;i++){
    int N=(int)N_list[i];
    int M=(int)(alpha*N);

    int Total_time=1500*N;
    int delta_time_memory=N/10;

    for (j=0;j<N_try_disorder_realization;j++){
      double ** J=(double **) malloc(M*sizeof(double *));

      int index_try=i*N_try_disorder_realization+j;
      double * mag=           (double *) malloc(Total_time/delta_time_memory*sizeof(double));
      double * time_rescale = (double *) malloc(Total_time/delta_time_memory*sizeof(double));


      printf("Size of the system:");
      printf("%d", N);
      printf("\n");
      printf("Try index:");
      printf("%d", index_try);
      printf("\n");

      setting_interaction(J,M,N_list[i],alpha,k_old);
      perceptron_quench(mag,time_rescale,J,M,N_list[i],index_try,alpha,k_old,k_new_list[i]);

      //// Write loop ////
      for (k=0;k<Total_time/delta_time_memory;k++){
        fprintf(fptr, "%d %f %f %f %f %f\n",index_try,N_list[i],k_old,k_new_list[i],time_rescale[k],mag[k]);
      }

      free( J );
      free( mag );
      free( time_rescale );
      printf("\n");
      printf("\n");
    }
  }




  fclose(fptr);
  return 0;
}
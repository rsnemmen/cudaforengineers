#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

#define ITERS_PER_RENDER 50 // number of iterations per render
#define W 640 // size of temperature array
#define H 640

float *d_temp = 0;
//int iterationCount = 0;
BC bc = {W / 2, H / 2, W / 10.f, 150, 212.f, 70.f, 0.f}; // Boundary conditions





// renders images with temperature distribution
void render(float *temp, int w, int h, BC bc) {
  // prints simulation parameters
  //char title[128];
  printf("Temperatures: T_s=%3.0f, T_a=%3.0f, T_g=%3.0f",
                  bc.t_s, bc.t_a, bc.t_g);
  //printf(title);

  // open file for writing
  FILE *f = fopen("output.dat", "w");
  if (f == NULL) {
      printf("Error opening file!\n");
      exit(1);
  }

  for (int i=0; i<w*h-1; i++) {
      fprintf(f, "%f\n", temp[i]);
  }

  fclose(f);
}






int main(int argc, char** argv) {
  cudaMalloc(&d_temp, W*H*sizeof(float));
  resetTemperature(d_temp, W, H, bc); // calls first kernel

  for (int i = 0; i < ITERS_PER_RENDER; ++i) {
    kernelLauncher(d_temp, W, H, bc); // calls kernel that solves PDE
  }

  float *temp = (float *)calloc(W*H, sizeof(float));
  cudaMemcpy(temp, d_temp, W*H*sizeof(float), cudaMemcpyDeviceToHost);  

  render(temp,W,H,bc);

  cudaFree(d_temp);
  free(temp);

  return 0;
}

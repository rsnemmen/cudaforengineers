#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#define ITERS_PER_RENDER 500 // number of iterations per render
#define W 640 // size of temperature array
#define H 640

BC bc = {W / 2, H / 2, W / 10.f, 150, 212.f, 51.f, 227.f}; // Boundary conditions





// renders images with temperature distribution
void render(float *temp, int w, int h, BC bc) {
  // prints simulation parameters
  //char title[128];
  printf("Temperatures: T_s=%3.0f, T_a=%3.0f, T_g=%3.0f \n",
                  bc.t_s, bc.t_a, bc.t_g);
  //printf(title);

  // open file for writing
  FILE *f = fopen("output.dat", "w");
  if (f == NULL) {
      printf("Error opening file!\n");
      exit(1);
  }

  for (int i=0; i<w*h; i++) {
      fprintf(f, "%f ", temp[i]);
  }

  fclose(f);
}






int main(int argc, char** argv) {
  float *d_temp = 0;

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

#include "interactions.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

#define ITERS_PER_RENDER 50 // number of iterations per render
#define NRENDERS 100 // number of images produced
#define W 640 // size of temperature array
#define H 640

float *d_temp = 0;
int iterationCount = 0;
BC bc = {W / 2, H / 2, W / 10.f, 150, 212.f, 70.f, 0.f}; // Boundary conds

// renders images with temperature distribution
void render(d_temp,W,H,bc) {
  // prints simulation parameters
  char title[128];
  sprintf(title, "Temperature Visualizer - Iterations=%4d, "
                  "T_s=%3.0f, T_a=%3.0f, T_g=%3.0f",
                  iterationCount, bc.t_s, bc.t_a, bc.t_g);
  printf(title);

  CODE FOR PRODUCING ARRAY
}


int main(int argc, char** argv) {
  cudaMalloc(&d_temp, W*H*sizeof(float));

  resetTemperature(d_temp, W, H, bc); // calls first kernel

  for (int i = 0; i < ITERS_PER_RENDER; ++i) {
    kernelLauncher(d_temp, W, H, bc);
  }

  render(d_temp,W,H,bc)

  cudaFree(d_temp);

  return 0;
}
#include <math.h>
#include <stdlib.h>
#define W 32
#define H 32
#define D 32

// struct that replaces the float3 from the CUDA coda
typedef struct {
  float x, y, z; 
} loc3d;

//__device__
float distance(int c, int r, int s, loc3d pos) {
  return sqrt((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y) +
               (s - pos.z)*(s - pos.z));
}

//__global__
void distanceKernel(float *d_out, int w, int h, int d, loc3d pos) {
  int i,c,r,s;

  for (c=0; c<w; c++) {
    for (r=0; r<h; r++) {
      for (s=0; s<d; s++) {
        i = c + r*w + s*w*h;
        d_out[i] = distance(c, r, s, pos); // compute and store result
      }
    }
  }
}

int main() {
  float *out = (float*)calloc(W*H*D, sizeof(float));

  loc3d pos = { 0.0f, 0.0f, 0.0f }; // set reference position

  distanceKernel(out, W, H, D, pos);

  free(out);
  return 0;
}

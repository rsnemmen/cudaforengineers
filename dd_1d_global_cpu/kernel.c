#include "kernel.h"

void ddParallel(float *out, const float *in, int n, float h) {  
  int i;

  for (i=0; i<n; i++) {
      out[i] = (in[i - 1] - 2.f*in[i] + in[i + 1]) / (h*h);
  }
}

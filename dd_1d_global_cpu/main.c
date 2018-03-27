#include "kernel.h"
#include <math.h>
#include <stdio.h>

int main() {
  const float PI = 3.1415927;
  const int N = 150000;
  const float h = 2 * PI / N;
  
  float x[N];
  float u[N];
  float result_parallel[N];

  for (int i = 0; i < N; ++i) {
    x[i] = 2 * PI*i / N;
    u[i] = sin(x[i]);
  }

  ddParallel(result_parallel, u, N, h);

  /*FILE *outfile = fopen("results.csv", "w");
  for (int i = 1; i < N - 1; ++i) {
    fprintf(outfile, "%f,%f,%f,%f\n", x[i], u[i],
            result_parallel[i], result_parallel[i] + u[i]);
  }
  fclose(outfile);*/
}

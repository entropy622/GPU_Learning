## CPU

# OpenMP
```
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
  for (int j = 0; j < n; ++j) {
    int sum = 0;
    for (int k = 0; k < n; ++k) {
      sum += a[i * n + k] * bt[j * n + k];
    }
    c[i * n + j] = sum;
  }
}
```
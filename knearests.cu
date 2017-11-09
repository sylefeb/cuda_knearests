// Sylvain Lefebvre 2017-10-04
#include <stdio.h>

#include <cuda_runtime.h>
#include <assert.h>

#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <random>

// ------------------------------------------------------------

#define KN_kernel        7
#define KN_global        28 // int * KN_kernel
#define POINTS_PER_BLOCK 64

// ------------------------------------------------------------

__device__ int cellFromPoint(int xdim, int ydim, int zdim, float x, float y, float z)
{
  int   i = (int)floor(x * (float)xdim);
  int   j = (int)floor(y * (float)ydim);
  int   k = (int)floor(z * (float)zdim);
  i = max(0, min(i, xdim - 1));
  j = max(0, min(j, ydim - 1));
  k = max(0, min(k, zdim - 1));
  return i + j*xdim + k*xdim*ydim;
}

__global__ void count(const float *points, int numPoints, int xdim, int ydim, int zdim, int *counters)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < numPoints) {
    float x = points[id * 3 + 0];
    float y = points[id * 3 + 1];
    float z = points[id * 3 + 2];
    int cell = cellFromPoint(xdim, ydim, zdim, x, y, z);
    atomicAdd(counters + cell, 1);
  }
}

__global__ void reserve(int xdim, int ydim, int zdim, const int *counters, int *globalcounter, int *ptrs)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < xdim*ydim*zdim) {
    int cnt = counters[id];
    if (cnt > 0) {
      ptrs[id] = 1 + atomicAdd(globalcounter, cnt); // adding 1 so that null tags empty
    }
  }
}

__global__ void store(const float *points, int numPoints, int xdim, int ydim, int zdim, const int *ptrs, int *counters, int num_stored, float *stored_points)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < numPoints) {
    float x = points[id * 3 + 0];
    float y = points[id * 3 + 1];
    float z = points[id * 3 + 2];
    int cell = cellFromPoint(xdim, ydim, zdim, x, y, z);
    int pos = ptrs[cell] + atomicAdd(counters + cell, 1);
    stored_points[pos * 3 + 0] = x;
    stored_points[pos * 3 + 1] = y;
    stored_points[pos * 3 + 2] = z;
  }
}

// Launch one per page, grouped by page size
__global__ void knearest(
  int xdim, int ydim, int zdim,
  int num_stored, const int *ptrs, const int *counters, const float *stored_points,
  int num_cell_offsets, const int *cell_offsets, const float *cell_offset_distances,
  unsigned int *g_knearests
  )
{
  float previous_max_dist = -1.0f;
  int   previous_max_id = -1;

  int point_in = 1 + threadIdx.x + blockIdx.x * POINTS_PER_BLOCK;

  if (point_in >= num_stored) return;

  // point considered by this thread
  float x = stored_points[point_in * 3 + 0];
  float y = stored_points[point_in * 3 + 1];
  float z = stored_points[point_in * 3 + 2];

  // bool empty_point = (x == 0.0f && y == 0.0f && z == 0.0f); // no longer possible as 'page size == 1'

  int cell_in = cellFromPoint(xdim, ydim, zdim, x, y, z);

  for (int knpass = 0; knpass < KN_global / KN_kernel; knpass++) {

    // each thread updates its k-nearests
#if 1
    int offs = threadIdx.x * KN_kernel;
    __shared__ unsigned int knearests[KN_kernel * POINTS_PER_BLOCK];
    __shared__ float        knearests_dists[KN_kernel * POINTS_PER_BLOCK];
#else
    const int offs = 0;
    unsigned int knearests      [KN_kernel];
    float        knearests_dists[KN_kernel];
#endif
    for (int i = 0; i < KN_kernel; i++) {
      knearests[offs + i] = UINT_MAX;
      knearests_dists[offs + i] = FLT_MAX;
    }
    int   knearests_prev_max_k = 0;
    float knearests_prev_max_d = FLT_MAX;
    int   knearests_prev_max_id = INT_MAX;

    for (int o = 0; o < num_cell_offsets; o++)  {

      float min_dist = cell_offset_distances[o];
      bool kn_completed = (knearests_prev_max_d < min_dist);

      if (kn_completed) break;

      int cell = cell_in + cell_offsets[o];
      if (cell >= 0 && cell < xdim*ydim*zdim) {

        int cell_base = ptrs[cell];

        int num = counters[cell];
        // go through all pages of this cell
        for (int ptr = cell_base; ptr < cell_base + num; ptr++) {

          // read point
          float x_cmp = stored_points[ptr * 3 + 0];
          float y_cmp = stored_points[ptr * 3 + 1];
          float z_cmp = stored_points[ptr * 3 + 2];

          float d = (x_cmp - x)*(x_cmp - x) + (y_cmp - y)*(y_cmp - y) + (z_cmp - z)*(z_cmp - z);

          if ((d < knearests_prev_max_d || (d == knearests_prev_max_d && ptr < knearests_prev_max_id))
            && (d > previous_max_dist || (d == previous_max_dist    && ptr > previous_max_id))) {
            // replace current max
            knearests[offs + knearests_prev_max_k] = ptr;
            knearests_dists[offs + knearests_prev_max_k] = d;
            // find out new max
            knearests_prev_max_d = -1.0f;
            knearests_prev_max_id = -1;
            for (int k = 0; k < KN_kernel; k++) {
              if (knearests_dists[offs + k] > knearests_prev_max_d
                || (knearests_dists[offs + k] == knearests_prev_max_d && knearests[offs + k] > knearests_prev_max_id)) {
                knearests_prev_max_k = k;
                knearests_prev_max_d = knearests_dists[offs + k];
                knearests_prev_max_id = knearests[offs + k];
              }
            }
          }
        }

      } // valid cell id
    } // offsets

    // store result
    for (int i = 0; i < KN_kernel; i++) {
      g_knearests[point_in + (knpass * KN_kernel + i) * num_stored] = knearests[offs + i];
    }
    previous_max_dist = knearests_dists[offs + knearests_prev_max_k];
    previous_max_id = knearests[offs + knearests_prev_max_k];
  }
}

// ------------------------------------------------------------

typedef struct {
  int K;
  int dimx, dimy, dimz;
  int num_cell_offsets;
  int allocated_points;
  int *d_cell_offsets;
  float *d_cell_offset_dists;
  int *d_counters;
  int *d_ptrs;
  int *d_globcounter;
  float *d_stored_points;
  unsigned int *d_knearests;
} kn_problem;

// ------------------------------------------------------------

void kn_firstbuild(kn_problem *kn,float *d_points, int numpoints)
{
  cudaError_t err = cudaSuccess;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // count points per grid cell
  {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numpoints + threadsPerBlock - 1) / threadsPerBlock;
    count << <blocksPerGrid, threadsPerBlock >> >(d_points, numpoints, kn->dimx, kn->dimy, kn->dimz, kn->d_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  // reserve memory for stored points
  {
    int threadsPerBlock = 256;
    int blocksPerGrid = (kn->dimx*kn->dimy*kn->dimz + threadsPerBlock - 1) / threadsPerBlock;
    reserve << <blocksPerGrid, threadsPerBlock >> >(kn->dimx, kn->dimy, kn->dimz, kn->d_counters, kn->d_globcounter, kn->d_ptrs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed (2) (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

#if 0
  // count requied point storage
  int globcounter = 0;
  cudaMemcpy(&globcounter, kn->d_globcounter, sizeof(int), cudaMemcpyDeviceToHost);
  fprintf(stderr, "max storage position: %d (%d points allocated) \n", 1 + globcounter, kn->allocated_points);
#endif

  // store
  {
    // zero counters
    cudaMemset(kn->d_counters, 0x00, kn->dimx*kn->dimy*kn->dimz*sizeof(int));
    // call kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numpoints + threadsPerBlock - 1) / threadsPerBlock;
    store << <blocksPerGrid, threadsPerBlock >> >(d_points, numpoints, kn->dimx, kn->dimy, kn->dimz, kn->d_ptrs, kn->d_counters, kn->allocated_points, kn->d_stored_points);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed (3) (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  fprintf(stderr, "kn_firstbuild: %f msec\n", milliseconds);
}

// ------------------------------------------------------------

kn_problem *kn_prepare(float *points, int numpoints)
{
  cudaError_t err = cudaSuccess;

  kn_problem *kn = (kn_problem*)malloc(sizeof(kn_problem));
  kn->K = KN_global;

  int sz = max(1,(int)round(pow(numpoints / 11.3f, 1.0f / 3.0)));
  kn->dimx = sz;
  kn->dimy = sz;
  kn->dimz = sz;

  // create cell offsets
  int Nmax = 8;
  if (sz < Nmax) {
    fprintf(stderr, "The current implementation does not support low number of input points\n");
    exit(EXIT_FAILURE);
  }
  // very naive approach, that's all fine, pre-computed once
  int alloc = Nmax*Nmax*Nmax*Nmax;
  int   *cell_offsets = (int*)malloc(alloc*sizeof(int));
  float *cell_offset_dists = (float*)malloc(alloc*sizeof(float));
  cell_offsets[0] = 0;
  cell_offset_dists[0] = 0.0f;
  kn->num_cell_offsets = 1;
  Nmax = min(sz, Nmax);
  for (int ring = 1; ring < Nmax; ring++) {
    for (int k = -Nmax / 2; k <= Nmax / 2; k++) {
      for (int j = -Nmax / 2; j <= Nmax / 2; j++) {
        for (int i = -Nmax / 2; i <= Nmax / 2; i++) {
          if (max(abs(i), max(abs(j), abs(k))) == ring) {
            int id_offset = i + j*kn->dimx + k*kn->dimx*kn->dimy;
            if (id_offset == 0) { 
              fprintf(stderr, "Error generating offsets\n");
              exit(EXIT_FAILURE); 
            }
            cell_offsets[kn->num_cell_offsets] = id_offset;
            float d = (float)(ring - 1) / (float)max(kn->dimx, max(kn->dimy, kn->dimz));
            cell_offset_dists[kn->num_cell_offsets] = d*d; // squared
            kn->num_cell_offsets++;
            if (kn->num_cell_offsets >= alloc) {
              exit(EXIT_FAILURE);
            }
          }
        }
      }
    }
  }

  // allocate cell offsets
  kn->d_cell_offsets = NULL;
  err = cudaMalloc((void **)&kn->d_cell_offsets, kn->num_cell_offsets*sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (1) to allocate (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(kn->d_cell_offsets, cell_offsets, kn->num_cell_offsets*sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (2) to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  free(cell_offsets);

  // allocate cell offset distances
  kn->d_cell_offset_dists = NULL;
  err = cudaMalloc((void **)&kn->d_cell_offset_dists, kn->num_cell_offsets*sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (3) to allocate (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(kn->d_cell_offset_dists, cell_offset_dists, kn->num_cell_offsets*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (4) to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  free(cell_offset_dists);

  // allocate input points
  float *d_points = NULL;
  err = cudaMalloc((void **)&d_points, numpoints * 3 * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (5) to allocate (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_points, points, numpoints * 3 * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (6) to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // allocate cell counters
  kn->d_counters = NULL;
  err = cudaMalloc((void **)&kn->d_counters, kn->dimx*kn->dimy*kn->dimz*sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (7) to allocate (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaMemset(kn->d_counters, 0x00, kn->dimx*kn->dimy*kn->dimz*sizeof(int));

  // allocate cell start pointers
  kn->d_ptrs = NULL;
  err = cudaMalloc((void **)&kn->d_ptrs, kn->dimx*kn->dimy*kn->dimz*sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (8) to allocate (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaMemset(kn->d_ptrs, 0x00, kn->dimx*kn->dimy*kn->dimz*sizeof(int));

  // allocate global counter for allocation
  kn->d_globcounter = NULL;
  err = cudaMalloc((void **)&kn->d_globcounter, sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (9) to allocate (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaMemset(kn->d_globcounter, 0x00, sizeof(int));

  // allocate stored points
  kn->allocated_points = numpoints + 1;
  kn->d_stored_points = NULL;
  err = cudaMalloc((void **)&kn->d_stored_points, kn->allocated_points * sizeof(float) * 3);
  cudaMemset(kn->d_stored_points, 0x00, kn->allocated_points * sizeof(float) * 3);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (10) to allocate (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // allocate knearests
  kn->d_knearests = NULL;
  err = cudaMalloc((void **)&kn->d_knearests, kn->allocated_points * KN_global * sizeof(int));
  cudaMemset(kn->d_knearests, 0xFF, kn->allocated_points * KN_global * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (11) to allocate (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // construct initial structure
  kn_firstbuild(kn,d_points,numpoints);

  // we no longer need the initial points
  cudaFree(d_points);

  return kn;
}

// ------------------------------------------------------------

void kn_solve(kn_problem *kn)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaError_t err = cudaSuccess;

  int threadsPerBlock = POINTS_PER_BLOCK;
  int blocksPerGrid = (kn->allocated_points + threadsPerBlock - 1) / POINTS_PER_BLOCK;

  cudaEventRecord(start);

  knearest << <blocksPerGrid, threadsPerBlock >> >(
    kn->dimx, kn->dimy, kn->dimz, kn->allocated_points,
    kn->d_ptrs, kn->d_counters, kn->d_stored_points,
    kn->num_cell_offsets, kn->d_cell_offsets, kn->d_cell_offset_dists,
    kn->d_knearests);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed  (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  fprintf(stderr,"kn_solve: %f msec\n", milliseconds);
}

// ------------------------------------------------------------

void kn_free(kn_problem **kn)
{
  cudaFree((*kn)->d_cell_offsets);
  cudaFree((*kn)->d_cell_offset_dists);
  cudaFree((*kn)->d_counters);
  cudaFree((*kn)->d_ptrs);
  cudaFree((*kn)->d_globcounter);
  cudaFree((*kn)->d_stored_points);
  cudaFree((*kn)->d_knearests);
  free(*kn);
  *kn = NULL;
}

// ------------------------------------------------------------

float *kn_get_points(kn_problem *kn)
{
  float *stored_points = (float*)malloc(kn->allocated_points * sizeof(float) * 3);
  cudaError_t err = cudaMemcpy(stored_points, kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[kn_get_points] Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return stored_points;
}

// ------------------------------------------------------------

unsigned int *kn_get_knearests(kn_problem *kn)
{
  unsigned int *knearests = (unsigned int*)malloc(kn->allocated_points * KN_global * sizeof(int));
  cudaError_t err = cudaMemcpy(knearests, kn->d_knearests, kn->allocated_points * KN_global * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[kn_get_knearests] Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return knearests;
}

// ------------------------------------------------------------

void kn_sanity_check(kn_problem *kn)
{
  cudaError_t err = cudaSuccess;

  float *stored_points = (float*)malloc(kn->allocated_points * sizeof(float) * 3);
  err = cudaMemcpy(stored_points, kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[kn_sanity_check:1] Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  unsigned int *knearests = (unsigned int*)malloc(kn->allocated_points * KN_global * sizeof(int));
  err = cudaMemcpy(knearests, kn->d_knearests, kn->allocated_points * KN_global * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[kn_sanity_check:2] Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int *counters = (int*)malloc(kn->dimx*kn->dimy*kn->dimz*sizeof(int));
  err = cudaMemcpy(counters, kn->d_counters, kn->dimx*kn->dimy*kn->dimz*sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[kn_sanity_check:3] Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int *ptrs = (int*)malloc(kn->dimx*kn->dimy*kn->dimz*sizeof(int));
  err = cudaMemcpy(ptrs, kn->d_ptrs, kn->dimx*kn->dimy*kn->dimz*sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[kn_sanity_check:4] Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // stats on counters
  int tot = 0;
  int cmin = INT_MAX, cmax = 0;
  std::map<int, int> histo;
  for (int c = 0; c < kn->dimx*kn->dimy*kn->dimz; c++) {
    histo[counters[c]]++;
    cmin = min(cmin, counters[c]);
    cmax = max(cmax, counters[c]);
    tot += counters[c];
  }
  printf("Grid:  points per cell: %d (min), %d (max), %f avg, total %d\n", cmin, cmax, (kn->allocated_points-1) / (float)(kn->dimx*kn->dimy*kn->dimz), tot);
  for (auto H : histo) {
    fprintf(stderr, "[%d] => %d\n", H.first, H.second);
  }

  std::minstd_rand rnd;
  int r = rnd();
  for (int tests = 0; tests < kn->allocated_points-1; tests++) {
    int allp = 1+((tests + r) % (kn->allocated_points-1));
  // for (int allp = 1; allp < kn->allocated_points; allp++) {
    //for (int allp = kn->allocated_points - 1; allp >= 0; allp--) {
    fprintf(stderr, "%d/%d ", allp, kn->allocated_points);
    float x = stored_points[allp * 3 + 0], y = stored_points[allp * 3 + 1], z = stored_points[allp * 3 + 2];
    //if (x == 0.0f && y == 0.0f && z == 0.0f) continue;
    // sanity check
    std::set<int> kns;
    for (int i = 0; i < KN_global; ++i) {
      int kni = knearests[allp + i*kn->allocated_points];
      if (kni < UINT_MAX) {
        if (kns.find(kni) != kns.end()) {
          for (int i = 0; i < KN_global; i++) {
            int kni = knearests[allp + i*kn->allocated_points];
            if (kni < UINT_MAX) {
              float kx = stored_points[kni * 3 + 0];
              float ky = stored_points[kni * 3 + 1];
              float kz = stored_points[kni * 3 + 2];
              int ci = (int)floor(kx*kn->dimx);
              int cj = (int)floor(ky*kn->dimy);
              int ck = (int)floor(kz*kn->dimz);
              float d = (x - kx)*(x - kx) + (y - ky)*(y - ky) + (z - kz)*(z - kz);
              fprintf(stderr, "   (%d) %d (%f,%f,%f) [%d,%d,%d] \t=> %f\n", i, kni, kx, ky, kz, ci, cj, ck, d);
            }
          }
          fprintf(stderr, "ERROR duplicated entry %d\n", kni);
          exit(EXIT_FAILURE);
        }
        kns.insert(kni);
      }
    }
    // now brute force search
    std::vector<std::pair<float, int> > kn_check;
    for (int c = 1; c < kn->allocated_points; c++) {
      float kx = stored_points[c * 3 + 0];
      float ky = stored_points[c * 3 + 1];
      float kz = stored_points[c * 3 + 2];
      // if (kx == 0.0f && ky == 0.0f && kz == 0.0f) continue;
      float d = (x - kx)*(x - kx) + (y - ky)*(y - ky) + (z - kz)*(z - kz);
      kn_check.push_back(std::make_pair(d, c));
      if (kn_check.size() > 100000) {
        std::sort(kn_check.begin(), kn_check.end());
        kn_check.resize(KN_global);
      }
    }
    std::sort(kn_check.begin(), kn_check.end());
    kn_check.resize(min((int)kn_check.size(),KN_global));
    for (int k = 0; k < (int)kn_check.size(); k++) {
      float kx = stored_points[kn_check[k].second * 3 + 0];
      float ky = stored_points[kn_check[k].second * 3 + 1];
      float kz = stored_points[kn_check[k].second * 3 + 2];
      int ci = (int)floor(kx*kn->dimx);
      int cj = (int)floor(ky*kn->dimy);
      int ck = (int)floor(kz*kn->dimz);
      if (kns.find(kn_check[k].second) == kns.end()) {
        // dump current configuration
        int pi = (int)floor(x*kn->dimx);
        int pj = (int)floor(y*kn->dimy);
        int pk = (int)floor(z*kn->dimz);
        fprintf(stderr, "============== (%f,%f,%f) [%d,%d,%d] (counter:%d ptr:%d)\n",
          x, y, z,
          pi, pj, pk,
          counters[pi + pj*kn->dimx + pk*kn->dimx*kn->dimy],
          ptrs[pi + pj*kn->dimx + pk*kn->dimx*kn->dimy]);
        for (int i = 0; i < KN_global; i++) {
          int kni = knearests[allp + i*kn->allocated_points];
          if (kni < UINT_MAX) {
            float kx = stored_points[kni * 3 + 0];
            float ky = stored_points[kni * 3 + 1];
            float kz = stored_points[kni * 3 + 2];
            int ci = (int)floor(kx*kn->dimx);
            int cj = (int)floor(ky*kn->dimy);
            int ck = (int)floor(kz*kn->dimz);
            float d = (x - kx)*(x - kx) + (y - ky)*(y - ky) + (z - kz)*(z - kz);
            fprintf(stderr, "   (%d) %d (%f,%f,%f) [%d,%d,%d] \t=> %f\n", i, kni, kx, ky, kz, ci, cj, ck, d);
          }
        }
        fprintf(stderr, "ERROR cannot find knearest %d SANITY CHECK FAILED\n", kn_check[k].second);
        float d = (x - kx)*(x - kx) + (y - ky)*(y - ky) + (z - kz)*(z - kz);
        fprintf(stderr, "**** [%d] (%f,%f,%f) [%d,%d,%d]  %d => %f (%f)\n", k, kx, ky, kz, ci, cj, ck, kn_check[k].second, kn_check[k].first, d);
        exit(EXIT_FAILURE);
      }
    }
    fprintf(stderr, " [ok]\n");
  }

  free(knearests);
  free(ptrs);
  free(counters);
  free(stored_points);

}

// ------------------------------------------------------------

int kn_num_points(kn_problem *kn)
{
  return kn->allocated_points - 1;
}


// ------------------------------------------------------------

typedef struct {
  float        *points;
  unsigned int *kns;
  int           K;
  int           allocated_points;
  int           point_id;
  int           k_rank;
} kn_iterator;

// ------------------------------------------------------------

kn_iterator  *kn_begin_enum(kn_problem *kn)
{
  kn_iterator *it = (kn_iterator*)malloc(sizeof(kn_iterator));
  it->K = kn->K;
  it->points = kn_get_points(kn);
  it->kns = kn_get_knearests(kn);
  it->allocated_points = kn->allocated_points;
  it->k_rank = -1;
  it->point_id = -1;
  return it;
}

// ------------------------------------------------------------

float        *kn_point(kn_iterator *it,int point_id)
{
  return it->points + (point_id+1)*3;
}

// ------------------------------------------------------------

float        *kn_first_nearest(kn_iterator *it, int point_id)
{
  it->point_id = point_id + 1;
  it->k_rank = 0;
  unsigned int kid = it->kns[it->point_id + it->k_rank*it->allocated_points];
  if (kid < UINT_MAX) {
    return it->points + kid*3;
  } else {
    return NULL;
  }
}

// ------------------------------------------------------------

float        *kn_next_nearest(kn_iterator *it)
{
  it->k_rank++;
  if (it->k_rank >= it->K) return NULL;
  unsigned int kid = it->kns[it->point_id + it->k_rank*it->allocated_points];
  if (kid < UINT_MAX) {
    return it->points + kid * 3;
  } else {
    return NULL;
  }
}

// ------------------------------------------------------------

void          kn_end_enum(kn_iterator **it)
{
  free((*it)->points);
  free((*it)->kns);
  free(*it);
  *it = NULL;
}

// ------------------------------------------------------------

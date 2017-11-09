// ------------------------------------------------------------

#include "knearests.h"

#include <random>

// ------------------------------------------------------------

int main(void)
{
  const int numPoints = 10000; 

  // random points
  std::minstd_rand rnd;
  float *in_points = (float *)malloc(numPoints * 3 * sizeof(float));
  for (int i = 0; i < numPoints; i++) {
    in_points[i * 3 + 0] = (float)(rnd()) / (float)(rnd.max());
    in_points[i * 3 + 1] = (float)(rnd()) / (float)(rnd.max());
    in_points[i * 3 + 2] = (float)(rnd()) / (float)(rnd.max());
  }

  kn_problem *kn = kn_prepare(in_points, numPoints);
  
  free(in_points);
  
  kn_solve(kn);

  // iterator, just show the few first points
  kn_iterator *it = kn_begin_enum(kn);
  for (int p = 0; p < min(3,kn_num_points(kn)); p++) {
    float *pt = kn_point(it, p);
    fprintf(stderr, "point %d (%f,%f,%f)\n",p,pt[0],pt[1],pt[2]);
    float *knpt = kn_first_nearest(it,p);
    int k = 0;
    while (knpt) {
      fprintf(stderr, "   knearest [%d] (%f,%f,%f)\n", k, knpt[0], knpt[1], knpt[2]);
      k++;
      knpt = kn_next_nearest(it);
    }
  }

  // kn_sanity_check(kn); // very slow sanity checks

  kn_free(&kn);

  return 0;
}

// ------------------------------------------------------------

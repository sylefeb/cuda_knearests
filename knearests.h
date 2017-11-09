// Sylvain Lefebvre 2017-10-04
#pragma once

struct kn_problem;
struct kn_iterator;

kn_problem   *kn_prepare(float *points, int numpoints);
void          kn_solve(kn_problem *kn);
void          kn_free(kn_problem **kn);
void          kn_sanity_check(kn_problem *kn);

float        *kn_get_points(kn_problem *kn);
unsigned int *kn_get_knearests(kn_problem *kn);

int           kn_num_points(kn_problem *kn);

kn_iterator  *kn_begin_enum(kn_problem *kn);
float        *kn_point(kn_iterator *iter, int point_id);
float        *kn_first_nearest(kn_iterator *iter,int point_id);
float        *kn_next_nearest (kn_iterator *iter);
void          kn_end_enum(kn_iterator **kn);

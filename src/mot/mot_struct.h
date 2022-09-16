#ifndef MOT_STRUCT_H
#define MOT_STRUCT_H
#include "tracker.h"

struct MOTTrackerList
{
    tracker *trackers = nullptr;
    size_t num_trackers;
};


#endif
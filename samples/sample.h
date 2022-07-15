#ifndef SAMPLE_H
#define SAMPLE_H
#include <gst/gst.h>
#include <glib.h>
#include <gst/gstbuffer.h>
#include <gst/gstpad.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

class Sample
{
public:
    Sample();
    ~Sample();

public:
    gint m_frame_number;
    
};
int test1(int argc, char **argv);
int test2(int argc, char **argv);
int test_mp4(int argc, char **argv);
int test_h265(int argc, char **argv);


#endif
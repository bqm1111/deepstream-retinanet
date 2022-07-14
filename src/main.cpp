#include <stdio.h>
#include <gst/gst.h>
#include <glib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../samples/sample.h"
#include "utils/VideoSource.h"

int main(int argc, char *argv[])
{
	test_mp4(argc, argv);
}

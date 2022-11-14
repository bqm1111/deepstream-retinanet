# Dependency library

Install dependency library: rapidjson, json-glib-1.0, eigen3
```bash
sudo apt-get update
sudo apt-get install -y rapidjson-dev libjson-glib-dev libeigen3-dev
```

# Convert model to .trt file

```bash
cd tools
sh convert.sh
```
# Run

```bash
cd build
cmake ..
make -j
./experiment <list_video_source_file> <is_live>
```

Run with video source file
```bash
./experiment ../configs/video_list.json 0
```
Run with rtsp stream 

```bash
./experiment ../configs/rtsp_source_list.json 1
```

Enable LATENCY mesurement

```bash
export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
```

# Check memory with LeakSanitizer
Add the following flag option to enable LeakSanitizer tool
```bash
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-omit-frame-pointer -fsanitize=leak -g -pthread")
```
Before running the program to check memory leak, remember to add the following enviroment variable

```bash
export LD_PRELOAD=<path-to-libgstnvfacealign.so>
```

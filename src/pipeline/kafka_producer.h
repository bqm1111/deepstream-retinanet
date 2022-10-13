#ifndef KAFKA_PRODUCER
#define KAFKA_PRODUCER
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <thread>
#include <librdkafka/rdkafkacpp.h>
#include "QDTLog.h"

class KafkaProducer
{
public:
    KafkaProducer();
    ~KafkaProducer();
    void init(std::string conn_str);
    RdKafka::Conf *conf;
    RdKafka::Producer *producer;
    int counter;
};
#endif
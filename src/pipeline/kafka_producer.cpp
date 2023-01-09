#include "kafka_producer.h"

KafkaProducer::KafkaProducer()
{
    counter = 0;
}

KafkaProducer::~KafkaProducer()
{
    delete producer;
    delete conf;
}

void KafkaProducer::init(std::string conn_str)
{
    conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    std::string errstr;
    if (conf->set("bootstrap.servers", conn_str.c_str(), errstr) !=
        RdKafka::Conf::CONF_OK)
    {
        QDTLog::error("{}", errstr);
        exit(1);
    }
    if (conf->set("message.max.bytes", "20485880", errstr) != RdKafka::Conf::CONF_OK)
    {
        QDTLog::error("{}", errstr);
        exit(1);
    }
    producer = RdKafka::Producer::create(conf, errstr);
    if (!producer)
    {
        QDTLog::error("Failed to create producer: {}", errstr);
        exit(1);
    }
}
#ifndef DEEPSTREAM_APP_CONFIG_H
#define DEEPSTREAM_APP_CONFIG_H
#include "ConfigBase.h"
#include <map>

enum class DSAppProperty
{
    FACE_PGIE_CONFIG,
    FACE_ALIGN_CONFIG,
    FACE_SGIE_CONFIG,
    FACE_FEATURE_CURL_ADDRESS,
    FACE_KAFKA_TOPIC,
    MOT_PGIE_CONFIG,
    MOT_SGIE_CONFIG,
    MOT_KAFKA_TOPIC,
    KAFKA_PROTO_LIB,
    KAFKA_CONNECTION_STR,
    KAFKA_MSG2P_LIB,
    MSG_CONFIG_PATH,
    MUXER_OUTPUT_WIDTH,
    MUXER_OUTPUT_HEIGHT,
    TILER_ROWS,
    TILER_COLS,
    TILER_WIDTH,
    TILER_HEIGHT
};

class DSAppConfig : public ConfigBase
{
private:
    std::map<DSAppProperty, std::string> DsAppDataField_ =
        {
            {DSAppProperty::FACE_PGIE_CONFIG, "FACE_PGIE_CONFIG"},
            {DSAppProperty::FACE_ALIGN_CONFIG, "FACE_ALIGN_CONFIG"},
            {DSAppProperty::FACE_SGIE_CONFIG, "FACE_SGIE_CONFIG"},
            {DSAppProperty::FACE_FEATURE_CURL_ADDRESS, "FACE_FEATURE_CURL_ADDRESS"},
            {DSAppProperty::FACE_KAFKA_TOPIC, "FACE_KAFKA_TOPIC"},
            {DSAppProperty::MOT_PGIE_CONFIG, "MOT_PGIE_CONFIG"},
            {DSAppProperty::MOT_SGIE_CONFIG, "MOT_SGIE_CONFIG"},
            {DSAppProperty::MOT_KAFKA_TOPIC, "MOT_KAFKA_TOPIC"},
            {DSAppProperty::KAFKA_PROTO_LIB, "KAFKA_PROTO_LIB"},
            {DSAppProperty::KAFKA_CONNECTION_STR, "KAFKA_CONNECTION_STR"},
            {DSAppProperty::KAFKA_MSG2P_LIB, "KAFKA_MSG2P_LIB"},
            {DSAppProperty::MSG_CONFIG_PATH, "MSG_CONFIG_PATH"},
            {DSAppProperty::MUXER_OUTPUT_WIDTH, "MUXER_OUTPUT_WIDTH"},
            {DSAppProperty::MUXER_OUTPUT_HEIGHT, "MUXER_OUTPUT_HEIGHT"},
            {DSAppProperty::TILER_ROWS, "TILER_ROWS"},
            {DSAppProperty::TILER_COLS, "TILER_COLS"},
            {DSAppProperty::TILER_WIDTH, "TILER_WIDTH"},
            {DSAppProperty::TILER_HEIGHT, "TILER_HEIGHT"}};

public:
    DSAppConfig(const std::string &name = "DSApp");
    ~DSAppConfig() override = default;

    /**
     * @brief Load all Property and its value from config file
     **/
    void load() override;

    /**
     * @brief Save new config to config file and change current
     * config data map
     * @return True if save successfully
     * Fail if save fail
     **/
    bool save() override;

    /**
     * @brief Clear all property dirty
     **/
    void clearDirty() override;

    /**
     * @return Vector of dirty properties
     **/
    std::vector<DirtyProperty> dirtyProperties() override;

    ConfigObj getProperty(DSAppProperty property);

    /**
     * @brief Get specific DSAppProperty
     **/
    template <typename T>
    T getProperty(DSAppProperty property)
    {
        if constexpr (std::is_same_v<T, std::string>)
        {
            return configDataMap_[property];
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            return std::stoi(configDataMap_[property]);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return std::stof(configDataMap_[property]);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return std::stod(configDataMap_[property]);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return ConfigHelper::string2Bool(configDataMap_[property]);
        }
    }

    /**
     * @brief Set specific DSAppProperty
     **/
    template <typename T>
    void setProperty(DSAppProperty property, T value)
    {
        if (!rootDataDoc_.child(DsAppDataField_[property].c_str()))
        {
            QDTLog::info("Adding {} to {} data field", DsAppDataField_[property], moduleConfigName_);
            rootDataDoc_.append_child(DsAppDataField_[property].c_str());
        }
        DsAppDirtyProperty_[property].oldValue_ = configDataMap_[property];
        if constexpr (std::is_same_v<T, std::string>)
        {
            DsAppDirtyProperty_[property].newValue_ = value;
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            if (value)
            {
                DsAppDirtyProperty_[property].newValue_ = "TRUE";
            }
            else
            {
                DsAppDirtyProperty_[property].newValue_ = "FALSE";
            }
        }
        else
        {
            DsAppDirtyProperty_[property].newValue_ = std::to_string(value);
        }
        if (DsAppDirtyProperty_[property].oldValue_ != DsAppDirtyProperty_[property].newValue_)
        {
            DsAppDirtyProperty_[property].dirty_ = true;
        }
        else
        {
            DsAppDirtyProperty_[property].dirty_ = false;
        }
        dirty_ = false;
        for (auto &[property, dirtyProperty] : DsAppDirtyProperty_)
        {
            if (dirtyProperty.dirty_)
            {
                dirty_ = true;
                break;
            }
        }
    }

private:
    std::map<DSAppProperty, std::string> configDataMap_;
    std::map<DSAppProperty, ConfigObj> configObjMap_;
    std::unordered_map<DSAppProperty, DirtyProperty> DsAppDirtyProperty_;
};

#endif
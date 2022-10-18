#ifndef DEEPSTREAM_APP_CONFIG_H
#define DEEPSTREAM_APP_CONFIG_H
#include "ConfigBase.h"
#include <map>

enum class DSAppProperty
{
    FACE_FEATURE_CURL_ADDRESS,
    KAFKA_METADATA_TOPIC,
    KAFKA_VISUAL_TOPIC,
    KAFKA_CONNECTION_STR,
    MUXER_OUTPUT_WIDTH,
    MUXER_OUTPUT_HEIGHT,
    TILER_ROWS,
    TILER_COLS,
    TILER_WIDTH,
    TILER_HEIGHT,
    FACE_CONFIDENCE_THRESHOLD,
    SAVE_CROP_IMG
};

class DSAppConfig : public ConfigBase
{
private:
    std::map<DSAppProperty, std::string> DsAppDataField_ =
        {
            {DSAppProperty::FACE_FEATURE_CURL_ADDRESS, "FACE_FEATURE_CURL_ADDRESS"},
            {DSAppProperty::KAFKA_METADATA_TOPIC, "KAFKA_METADATA_TOPIC"},
            {DSAppProperty::KAFKA_VISUAL_TOPIC, "KAFKA_VISUAL_TOPIC"},
            {DSAppProperty::KAFKA_CONNECTION_STR, "KAFKA_CONNECTION_STR"},
            {DSAppProperty::MUXER_OUTPUT_WIDTH, "MUXER_OUTPUT_WIDTH"},
            {DSAppProperty::MUXER_OUTPUT_HEIGHT, "MUXER_OUTPUT_HEIGHT"},
            {DSAppProperty::TILER_ROWS, "TILER_ROWS"},
            {DSAppProperty::TILER_COLS, "TILER_COLS"},
            {DSAppProperty::TILER_WIDTH, "TILER_WIDTH"},
            {DSAppProperty::TILER_HEIGHT, "TILER_HEIGHT"},
            {DSAppProperty::FACE_CONFIDENCE_THRESHOLD, "FACE_CONFIDENCE_THRESHOLD"},
            {DSAppProperty::SAVE_CROP_IMG, "SAVE_CROP_IMG"}};

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
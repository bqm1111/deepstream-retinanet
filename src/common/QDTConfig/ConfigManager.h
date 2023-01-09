#ifndef CONFIGMANAGER_H
#define CONFIGMANAGER_H
// #include <QDTGeneralManager.h>
#include <QDTLog.h>
#include <ConfigBase.h>
#include <unordered_map>
enum class ConfigType : unsigned char
{
    DeepStreamApp,
};

class ConfigManager
{
public:
    ConfigManager();
    // void setContext(QDTGeneralManager* generalManager) override;
    void setContext();

    std::shared_ptr<ConfigBase> getConfig(ConfigType configType);
    std::vector<std::pair<ConfigType, std::shared_ptr<ConfigBase>>> getDirtyConfig();
    void saveConfig();
    void clearDirtyConfig();

private:
    std::unordered_map<ConfigType, std::string> configName_ =
        {
            {ConfigType::DeepStreamApp, "DsApp"}};
    std::unordered_map<ConfigType, std::string> fileConfigMap_ =
        {
            {ConfigType::DeepStreamApp, "../configs/app.conf"}};
    std::unordered_map<ConfigType, std::shared_ptr<ConfigBase>> configMap_;
    std::mutex mutexDocument_;
};

#endif // CONFIGMANAGER_H

#include "ConfigManager.h"

ConfigManager::ConfigManager()
{
    for(auto&[configType, config] : fileConfigMap_)
    {
        configMap_[configType] = std::make_shared<ConfigBase>(configName_[configType]);
    }
}

std::shared_ptr<ConfigBase> ConfigManager::getConfig(ConfigType configType)
{
    if(configMap_.count(configType) == 1)
    {
        return configMap_[configType];
    }
    return nullptr;
}

std::vector<std::pair<ConfigType, std::shared_ptr<ConfigBase>>> ConfigManager::getDirtyConfig()
{
    std::vector<std::pair<ConfigType, std::shared_ptr<ConfigBase>>> dirtyConfig;
    for(auto&[configType, config] : configMap_)
    {
        if(config->isDirty())
        {
            dirtyConfig.push_back(std::make_pair(configType, config));
        }
    }
    return dirtyConfig;
}

void ConfigManager::saveConfig()
{
    for(auto&[configType, config] : configMap_)
    {
        if(config->isDirty())
        {
            config->save();
        }
    }
}

void ConfigManager::clearDirtyConfig()
{
    for(auto&[configType, config] : configMap_)
    {
        config->clearDirty();
    }
}

void ConfigManager::setContext()
{
    // QDTManager::setContext(generalManager);
    for(auto&[configType, config] : configMap_)
    {
        config->loadFile(fileConfigMap_[configType]);
        config = ConfigFactory::createConfig(config->cType(), config->modulName());
        if(config == nullptr){
            printf("nullptr\n");
        }
        config->loadFile(fileConfigMap_[configType]);
        config->load();
    }
}


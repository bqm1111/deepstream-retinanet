#include "ConfigBase.h"
#include "DeepStreamAppConfig.h"

ConfigBase::ConfigBase(const std::string &name) : moduleConfigName_(name)
{
}

bool ConfigBase::save()
{
    if (rootDataDoc_.document_element())
    {
        if (rootDataDoc_.save_file(fileConfigName_.c_str()))
        {
            QDTLog::debug("Saved {} config data to {}", moduleConfigName_, fileConfigName_);
        }
        else
        {
            QDTLog::debug("Failed saving config data to {}", fileConfigName_);
            return false;
        }
    }
    else
    {
        QDTLog::debug("Link ConfigManager Tree is empty. Nothing to save");
    }
    return true;
}

void ConfigBase::clearDirty()
{
    dirty_ = false;
}

bool ConfigBase::isDirty()
{
    return dirty_;
}

void ConfigBase::loadFile(const std::string &fileName)
{
    fileConfigName_ = fileName;
    QDTLog::info("Config file name = {}", fileName);
    if (!rootDataDoc_.load_file(fileName.c_str()))
    {
        QDTLog::error("Fail to load {} config file", moduleConfigName_);
    }
    else
    {
        QDTLog::debug("Successfully to load {} config file", moduleConfigName_);
        std::string cType = "TYPE";
        pugi::xml_node node = rootDataDoc_.child(cType.c_str());
        if (!node)
        {
            QDTLog::error("TYPE of config is not specified in {}", fileConfigName_);
            cType_ = CType::NotLink;
        }
        else
        {
            if (node.text().as_string() == cTypeStr_[CType::DeepStreamApp])
            {
                cType_ = CType::DeepStreamApp;
            }
        }
    }
}

CType ConfigBase::cType() const
{
    return cType_;
}

std::string ConfigBase::modulName() const
{
    return moduleConfigName_;
}

std::shared_ptr<ConfigBase> ConfigFactory::createConfig(CType configType, const std::string &modulName)
{
    switch (configType)
    {
    case CType::DeepStreamApp:
    {
        return std::make_shared<DSAppConfig>(modulName);
    }
    case CType::NotLink:
    {
        return nullptr;
    }
    default:
    {
        return std::make_shared<DSAppConfig>(modulName);
    }
    }
}

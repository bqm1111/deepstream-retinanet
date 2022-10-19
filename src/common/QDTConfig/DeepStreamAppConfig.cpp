#include "DeepStreamAppConfig.h"

DSAppConfig::DSAppConfig(const std::string &name) : ConfigBase(name)
{
}

void DSAppConfig::load()
{
    if (!rootDataDoc_)
    {
        QDTLog::debug("DsApp Config does not exist in linkConfig file");
    }
    else
    {
        for (auto &[prop, val] : DsAppDataField_)
        {
            pugi::xml_node node = rootDataDoc_.child(val.c_str());
            if(!node)
            {
                continue;
            }
            // current data
            configDataMap_[prop] = node.text().as_string();
            configObjMap_[prop] = ConfigObj(node.text().as_string());
            // Create dirty property data
            DirtyProperty dirtyProperty;
            dirtyProperty.oldValue_ = configDataMap_[prop];
            dirtyProperty.propertyName_ = val;
            DsAppDirtyProperty_[prop] = dirtyProperty;
        }
    }
    dirty_ = false;
}

bool DSAppConfig::save()
{
    for (auto &[property, dirtyProperty] : DsAppDirtyProperty_)
    {
        if (dirtyProperty.dirty_)
        {
            pugi::xml_node node = rootDataDoc_.child(DsAppDataField_[property].c_str());
            // change data of dirty property with newValue;
            configDataMap_[property] = DsAppDirtyProperty_[property].newValue_;
            configObjMap_[property] = ConfigObj(configDataMap_[property]);
            // replace newValue to file config
            node.text().set(DsAppDirtyProperty_[property].newValue_.c_str());
            dirtyProperty.dirty_ = false;
        }
    }
    if (rootDataDoc_.document_element())
    {
        if (rootDataDoc_.save_file(fileConfigName_.c_str()))
        {
            QDTLog::debug("[DSAppConfig] Saved {} config data to {}", moduleConfigName_, fileConfigName_);
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
    dirty_ = false;
    return true;
}

void DSAppConfig::clearDirty()
{
    for (auto &[property, dirtyProperty] : DsAppDirtyProperty_)
    {
        if (dirtyProperty.dirty_)
        {
            dirtyProperty.dirty_ = false;
        }
    }
    dirty_ = false;
}

std::vector<DirtyProperty> DSAppConfig::dirtyProperties()
{
    std::vector<DirtyProperty> dirtyVec;
    for (auto &[property, dirtyProperty] : DsAppDirtyProperty_)
    {
        if (dirtyProperty.dirty_)
        {
            dirtyVec.push_back(dirtyProperty);
        }
    }
    return dirtyVec;
}

ConfigObj DSAppConfig::getProperty(DSAppProperty property)
{
    return configObjMap_[property];
}

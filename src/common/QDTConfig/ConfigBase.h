#ifndef CONFIGBASE_H
#define CONFIGBASE_H
#include <string>
#include <atomic>
#include <pugixml.hpp>
#include <pugixml.hpp>
#include <ConfigHelper.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include "QDTLog.h"

enum class CType : unsigned char
{
    NotLink = 0,
    DeepStreamApp = 1
};

class ConfigObj
{
public:
    ConfigObj(){}
    ConfigObj(std::string const& data) : data_(data) {}
    int toInt()
    {
        return std::stoi(data_);
    }
    float toFloat()
    {
        return std::stof(data_);
    }
    bool toBool()
    {
        return  ConfigHelper::string2Bool(data_);
    }
    std::string toString()
    {
        return data_;
    }
private:
    std::string data_;
};

struct DirtyProperty
{
    std::string oldValue_;
    std::string newValue_;
    std::string propertyName_;
    bool dirty_ = false;
};

class ConfigBase
{
public:
    explicit ConfigBase(const std::string& name);
    virtual ~ConfigBase() {}
    virtual void load() {}
    virtual bool save();
    virtual void clearDirty();
    virtual std::vector<DirtyProperty> dirtyProperties(){ return std::vector<DirtyProperty>();}
    bool isDirty();
    void loadFile(std::string const& fileName);
    CType cType() const;
    std::string modulName() const;
private:
    std::unordered_map<CType, std::string> cTypeStr_ =
    {
        {CType::DeepStreamApp, "DeepStreamApp"}
    };
protected:
    std::string moduleConfigName_;
    std::string fileConfigName_;
    pugi::xml_document rootDataDoc_;
    CType cType_;
    std::atomic_bool dirty_{false};
};

class ConfigFactory
{
public:
    static std::shared_ptr<ConfigBase> createConfig(CType configType, const std::string& modulName);
};

#endif // CONFIGBASE_H

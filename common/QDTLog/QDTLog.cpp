#include "QDTLog.h"
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
QDTLog::QDTLog()
{

}

bool QDTLog::isDirExist(const std::string &_dir)
{
    struct stat info;
    if (stat(_dir.c_str(), &info) != 0)
    {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

bool QDTLog::makeDir(const std::string &_dir)
{
    mode_t mode = 0755;
    int ret = mkdir(_dir.c_str(), mode);
    if (ret == 0)
        return true;
    switch (errno)
    {
    case ENOENT: {
        int pos = _dir.find_last_of('/');
        if (pos == static_cast<int>(std::string::npos))
            return false;
        if (!makeDir( _dir.substr(0, pos) ))
            return false;
        return 0 == mkdir(_dir.c_str(), mode);
    }
    case EEXIST:
        return isDirExist(_dir);

    default:
        return false;
    }
}

std::string QDTLog::getCurrLocalDateTime(const std::string &_fmt)
{
    char res[80];
    time_t t = time(0);
    struct tm *tmp = localtime(&t);
    if (tmp == NULL) return "";
    if (strftime(res, sizeof(res), _fmt.c_str(), tmp) == 0) return "";
    return std::string(res);
}

bool QDTLog::init(const std::string& folderName, QDTLogLevel level, float size, int duplicate)
{
    std::lock_guard<std::mutex> locker(mutexMap_);
    if(!initFlag_)
    {
        folderName_ = folderName;
        size_ = size;
        duplicate_ = duplicate;
        if(!isDirExist(folderName_)){
            if(!makeDir(folderName_)){
                return false;
            }
        }
        auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto rotatingFileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(folderName+"/"+getCurrLocalDateTime()+".log", bit2Mb_*size, duplicate);
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(consoleSink);
        sinks.push_back(rotatingFileSink);
        logMap_["root"] = std::make_shared<spdlog::logger>("root", begin(sinks), end(sinks));
        rootLog_ = logMap_["root"];
        rootLog_->set_pattern(pattern_);
        rootLog_->set_level(static_cast<spdlog::level::level_enum>(level));
        return true;
    }
    return true;
}

std::shared_ptr<spdlog::logger> QDTLog::get(const std::string& loggerName)
{
    std::lock_guard<std::mutex> locker(mutexMap_);
    if(logMap_.count(loggerName) == 0)
    {
        auto fileLog = spdlog::rotating_logger_mt(loggerName, folderName_+"/"+loggerName+".log", bit2Mb_*size_, duplicate_);
        logMap_[loggerName] = spdlog::get(loggerName);
        logMap_[loggerName]->set_pattern("%v");
        logMap_[loggerName]->set_level(spdlog::level::trace);
        return logMap_[loggerName];
    }
    else{
        return logMap_[loggerName];
    }
}

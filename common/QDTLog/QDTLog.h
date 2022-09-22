#ifndef QDTLOG_H
#define QDTLOG_H
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bin_to_hex.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <unordered_map>
#include <iostream>
#include <mutex>
#include <sstream>

enum QDTLogLevel
{
    Trace = SPDLOG_LEVEL_TRACE,
    Debug = SPDLOG_LEVEL_DEBUG,
    Info = SPDLOG_LEVEL_INFO,
    Warn = SPDLOG_LEVEL_WARN,
    Error = SPDLOG_LEVEL_ERROR,
    Critical = SPDLOG_LEVEL_CRITICAL,
    Off = SPDLOG_LEVEL_OFF,
    n_levels
};

class SourceLoc
{
public:
    SourceLoc(const std::string& file = __FILE__, int line = __LINE__)
        : file_(file), line_(line)
    {}
    std::string file_;
    int line_;
};

class QDTLog
{
public:    
    static bool init(const std::string& folderName = ".", QDTLogLevel level = QDTLogLevel::Debug, float size = 5, int duplicate = 2);
    static std::shared_ptr<spdlog::logger> get(const std::string& loggerName);

    template<typename... Args>
    static void trace(Args... args)
    {
        rootLog_->trace(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void debug(Args... args)
    {
        rootLog_->debug(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void info(Args... args)
    {
        rootLog_->info(std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    static void warn(T arg1, Args... args)
    {
        rootLog_->warn(arg1, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void error(Args... args)
    {
        rootLog_->error(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void critical(Args... args)
    {
        rootLog_->critical(std::forward<Args>(args)...);
    }

    // log for any logger
    template<typename... Args>
    static void trace(std::shared_ptr<spdlog::logger> logger, Args... args)
    {
        logger->trace(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void debug(std::shared_ptr<spdlog::logger> logger, Args... args)
    {
        logger->debug(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void info(std::shared_ptr<spdlog::logger> logger, Args... args)
    {
        logger->info(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void warn(std::shared_ptr<spdlog::logger> logger, Args... args)
    {
        logger->warn(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void error(std::shared_ptr<spdlog::logger> logger, Args... args)
    {
        logger->error(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void critical(std::shared_ptr<spdlog::logger> logger, Args... args)
    {
        logger->critical(std::forward<Args>(args)...);
    }

    // extra log ---- get file name and line
    template<typename... Args>
    static void extraTrace(SourceLoc SourceLoc, const std::string& first, Args... args)
    {
        std::stringstream head;
        head << "[" << SourceLoc.file_ << ":" << SourceLoc.line_ << "] " << first;
        rootLog_->trace(head.str(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void extraDebug(SourceLoc SourceLoc, const std::string& first, Args... args)
    {
        std::stringstream head;
        head << "[" << SourceLoc.file_ << ":" << SourceLoc.line_ << "] " << first;
        rootLog_->debug(head.str(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void extraInfo(SourceLoc SourceLoc, const std::string& first, Args... args)
    {
        std::stringstream head;
        head << "[" << SourceLoc.file_ << ":" << SourceLoc.line_ << "] " << first;
        rootLog_->info(head.str(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void extraWarn(SourceLoc SourceLoc, const std::string& first, Args... args)
    {
        std::stringstream head;
        head << "[" << SourceLoc.file_ << ":" << SourceLoc.line_ << "] " << first;
        rootLog_->warn(head.str(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void extraError(SourceLoc SourceLoc, const std::string& first, Args... args)
    {
        std::stringstream head;
        head << "[" << SourceLoc.file_ << ":" << SourceLoc.line_ << "] " << first;
        rootLog_->error(head.str(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void extraCritical(SourceLoc SourceLoc, const std::string& first, Args... args)
    {
        std::stringstream head;
        head << "[" << SourceLoc.file_ << ":" << SourceLoc.line_ << "] " << first;
        rootLog_->critical(head.str(), std::forward<Args>(args)...);
    }

private:
    QDTLog();
    static bool isDirExist(const std::string &_dir);
    static bool makeDir(const std::string &_dir);
    static std::string getCurrLocalDateTime(const std::string &_fmt = "%Y-%m-%d_%H-%M-%S");
private:
    static inline std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> logMap_;
    static inline std::shared_ptr<spdlog::logger> rootLog_;
    static inline std::string fileName_ = "rollingFile.txt";
    static inline int bit2Mb_ = 1048576;
    static inline bool initFlag_{false};
    static inline std::string folderName_;
    static inline int size_;
    static inline int duplicate_;
    static inline std::mutex mutexMap_;
    static inline std::string pattern_ = "%^[%H:%M:%S.%e] [%l] %v%$";
};

#define QDTLOG_TRACE(...) QDTLog::extraTrace(SourceLoc(__FILE__, __LINE__), __VA_ARGS__)
#define QDTLOG_DEBUG(...) QDTLog::extraDebug(SourceLoc(__FILE__, __LINE__), __VA_ARGS__)
#define QDTLOG_INFO(...) QDTLog::extraInfo(SourceLoc(__FILE__, __LINE__), __VA_ARGS__)
#define QDTLOG_WARN(...) QDTLog::extraWarn(SourceLoc(__FILE__, __LINE__), __VA_ARGS__)
#define QDTLOG_ERROR(...) QDTLog::extraError(SourceLoc(__FILE__, __LINE__), __VA_ARGS__)
#define QDTLOG_CRITICAL(...) QDTLog::extraCritical(SourceLoc(__FILE__, __LINE__), __VA_ARGS__)
#endif // QDTLOG_H

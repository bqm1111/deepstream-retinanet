#ifndef CONFIGHELPER_H
#define CONFIGHELPER_H

#include <iostream>

class ConfigHelper
{
public:
    static bool isIntNumber(const std::string& str)
    {
        std::string check;
        if(str[0] == '-')
        {
            check = str.substr(1, str.length());
        }
        else
        {
            check = str;
        }
        for (char const &c : check) {
            if (std::isdigit(c) == 0) return false;
        }
        return true;
    }

    static bool isFloatNumber(const std::string& str)
    {
        int cnt = 0;
        for (char const &c : str) {
            if (std::isdigit(c) == 0)
            {
                if(c == '.' || c== ',')
                    cnt++;
                else
                {
                    return false;
                }
            }
        }
        if(cnt > 1)
            return false;

        return true;
    }
    static bool string2Bool(std::string str)
    {
        char c = str.c_str()[0];
        if(c == 't' || c== 'T' || c=='1' || c=='y' || c == 'Y')
            return true;
        return false;
    }
};

#endif // CONFIGHELPER_H

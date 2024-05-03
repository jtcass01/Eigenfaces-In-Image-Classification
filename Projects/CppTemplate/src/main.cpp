#include <cstring>
#include <fstream>
#include <iostream>

#include "../inc/template.hpp"

void processFile(const std::string filename)
{
    std::ifstream file(filename, std::fstream::in);

    for (std::string line; std::getline(file, line);)
    {
        char* pLineChar = strcpy(new char[line.length() + 1], line.c_str());

        // Do something with pLineChar
    }
}

int main()
{
    Example example;
    std::cout << "Hello world!" << std::endl;
    return 0;
}
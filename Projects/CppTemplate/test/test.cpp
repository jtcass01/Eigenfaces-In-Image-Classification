/**
 * @author Jacob Taylor Cassady
*/

// C++ Standard Library
#include <iostream>

// 3rd Party Libraries
#include "gtest/gtest.h"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "RUNNING TESTS..." << std::endl;

    return RUN_ALL_TESTS();
}
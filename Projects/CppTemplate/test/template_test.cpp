#include <gtest/gtest.h>
#include "../inc/template.hpp"


namespace 
{

TEST(ExampleFunctionTest, PositiveNos) {
    EXPECT_EQ(2, ExampleFunction(1, 1));
    EXPECT_EQ(0, ExampleFunction(1, -1));
    EXPECT_EQ(1, ExampleFunction(1, 0));
    EXPECT_EQ(1, ExampleFunction(0, 1));
}

TEST(ExampleFunctionTest, NegativeNos) {
    EXPECT_EQ(-2, ExampleFunction(-1, -1));
    EXPECT_EQ(0, ExampleFunction(-1, 1));
    EXPECT_EQ(-1, ExampleFunction(-1, 0));
    EXPECT_EQ(-1, ExampleFunction(0, -1));
}

}


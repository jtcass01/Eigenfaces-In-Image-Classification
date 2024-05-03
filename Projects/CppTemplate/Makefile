# compiler
CXX = g++

# compile flags
CXXFLAGS = -std=c++17 -Wall -I /usr/include/gtest/ -c
LXXFLAGS = -std=c++17

# libs
LIBS = inc/template.hpp

GTEST = /usr/lib/libgtest.a

# source files
SRCS = src/template.cpp src/main.cpp
TEST_SRCS = src/template.cpp test/template_test.cpp test/test.cpp

# object files
OBJS = $(SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# target
TARGET = solution
TEST_TARGET = unit_test

all: $(TARGET)

# Generates the target executable
$(TARGET):$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

unit_test: $(TEST_TARGET)

$(TEST_TARGET):$(TEST_OBJS)
	$(CXX) -o $(TEST_TARGET) $(TEST_OBJS) $(LIBS) $(GTEST)

# Removes object files and target
clean:
	rm -f $(OBJS) $(TARG) *~
	rm -f unit_test *~
	rm -f solution  *~

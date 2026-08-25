# Compiler
NVCC = nvcc

# Compiler Flags
NVCCFLAGS = -std=c++17 -O3

# Executable Name
TARGET = canny_npp_example

# Source Files
SRC = src/canny_npp_example.cpp

# Libraries
LIBS = -lnppc -lnppif -lnppicc -lnppisu -lnppial -lcudart

# Default Rule
all: $(TARGET)

# Compile Executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) $(LIBS)

# Clean Rule
clean:
	rm -f $(TARGET)

.PHONY: all clean
# Compiler
NVCC = nvcc

# Compiler Flags
NVCCFLAGS = -std=c++17 -O2

# Executable Name
TARGET = note_detector

# Source Files
SRC = src/note_detector.cu

# Libraries
LIBS = -lcufft -lsndfile

# Default Rule
all: $(TARGET)

# Compile Executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) $(LIBS)

# Clean Rule
clean:
	rm -f $(TARGET)

.PHONY: all clean

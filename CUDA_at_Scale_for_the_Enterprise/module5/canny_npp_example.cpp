#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include <nppi.h>
#include <nppdefs.h>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Check for cuda errors
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err__) << std::endl;    \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

// Check for NPP errors
#define NPP_CHECK(call)                                                      \
    do {                                                                     \
        NppStatus st__ = (call);                                             \
        if (st__ != NPP_SUCCESS) {                                          \
            std::cerr << "NPP error at " << __FILE__ << ":" << __LINE__      \
                      << " - status code " << st__ << std::endl;             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)


// Check directorPath and get all the filenames
std::vector<std::string> getFilenames(const std::string& directoryPath) {

    std::vector<std::string> filenames;

    if (!std::filesystem::exists(directoryPath) || !std::filesystem::is_directory(directoryPath)) {
        std::cerr << "Error: \"" << directoryPath << "\" is not a valid directory."<<std::endl;
        return filenames;
    }

    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            filenames.push_back(directoryPath + entry.path().filename().string());
        }
    }

    return filenames;
}

// Inserts "_edges" before the extension, e.g. "example.jpg" -> "example_edges.jpg"
std::vector<std::string> addEdgesTag(const std::vector<std::string>& filenames, const std::string& directoryPath) {
    std::vector<std::string> edgeFilenames;
    edgeFilenames.reserve(filenames.size());

    for (const auto& name : filenames) {
        std::filesystem::path p(name);
        std::string stem = p.stem().string();
        std::string ext  = p.extension().string();

        edgeFilenames.push_back(directoryPath + stem + "_edges" + ext);
    }

    return edgeFilenames;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                   << " <input directory of jpgs> <output directory of jpgs>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];

    // Read the input filenames and create their corresponding output filenames
    std::vector<std::string> inputFilenames{getFilenames(inputPath)};
    std::vector<std::string> outputFilenames{addEdgesTag(inputFilenames, outputPath)};

    // Some values for the edge detection. We'll just hardcode these.
    const Npp16s lowThresh  = static_cast<Npp16s>(72);
    const Npp16s highThresh = static_cast<Npp16s>(256);

    // Loop over the input files and output the edges
    for(int i=0; i<inputFilenames.size(); i++)
    {
        std::string inputFilename = inputFilenames[i];
        std::string outputFilename = outputFilenames[i];

        // Decode the input image
        int width = 0, height = 0, srcChannels = 0;
        unsigned char* hostRGB =
            stbi_load(inputFilename.c_str(), &width, &height, &srcChannels, 3);
        if (!hostRGB) {
            std::cerr << "Failed to load image '" << inputFilename
                    << "': " << stbi_failure_reason() << std::endl;
            return EXIT_FAILURE;
        }

        NppiSize roiSize{ width, height };

        // Upload the image to device memory
        int rgbStep = 0;
        Npp8u* d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);
        if (!d_rgb) {
            std::cerr << "nppiMalloc_8u_C3 failed for RGB buffer" << std::endl;
            return EXIT_FAILURE;
        }

        CUDA_CHECK(cudaMemcpy2D(d_rgb, rgbStep,
                                hostRGB, width * 3,
                                width * 3, height,
                                cudaMemcpyHostToDevice));
        stbi_image_free(hostRGB);

        // Convert RGB -> 8-bit grayscale
        int graySrcStep = 0;
        Npp8u* d_gray = nppiMalloc_8u_C1(width, height, &graySrcStep);
        if (!d_gray) {
            std::cerr << "nppiMalloc_8u_C1 failed for grayscale buffer" << std::endl;
            return EXIT_FAILURE;
        }

        NPP_CHECK(nppiRGBToGray_8u_C3C1R(d_rgb, rgbStep,
                                        d_gray, graySrcStep,
                                        roiSize));
        nppiFree(d_rgb);

        // Allocate the Canny destination + scratch buffers
        int dstStep = 0;
        Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
        if (!d_dst) {
            std::cerr << "nppiMalloc_8u_C1 failed for destination buffer" << std::endl;
            return EXIT_FAILURE;
        }

        int scratchBytes = 0;  // We could probably calculate a max size and use that 
                               // instead of allocating/deallocating memory...
        NPP_CHECK(nppiFilterCannyBorderGetBufferSize(roiSize, &scratchBytes));

        Npp8u* d_scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scratch, scratchBytes));

        // Run the Canny edge detector on the grayscale image
        NppiPoint srcOffset{ 0, 0 };

        // Note I'm hardcoding a lot of values here...
        NPP_CHECK(nppiFilterCannyBorder_8u_C1R(
            d_gray, graySrcStep, roiSize, srcOffset,
            d_dst, dstStep, roiSize,
            NPP_FILTER_SOBEL,
            NPP_MASK_SIZE_3_X_3,
            lowThresh, highThresh,
            nppiNormL2,
            NPP_BORDER_REPLICATE,
            d_scratch));

        CUDA_CHECK(cudaDeviceSynchronize());

        // Send the result to the host
        std::vector<unsigned char> hostEdges(static_cast<size_t>(width) * height);
        CUDA_CHECK(cudaMemcpy2D(hostEdges.data(), width,
                                d_dst, dstStep,
                                width, height,
                                cudaMemcpyDeviceToHost));

        // Encode and save the jpg 
        int writeOk = 0;
        writeOk = stbi_write_jpg(outputFilename.c_str(), width, height, 1, hostEdges.data(), 95);

        if (!writeOk) {
            std::cerr << "Failed to write output image: " << outputFilename << std::endl;
            return EXIT_FAILURE;
        }

        // Cleanup memory
        cudaFree(d_scratch);
        nppiFree(d_gray);
        nppiFree(d_dst);
    }

    return EXIT_SUCCESS;
}

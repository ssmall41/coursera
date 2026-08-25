# Canny Edge Detection with NPP

This program uses NPP convert .jpg files to an image of their edges. The code was tested on [ImageNet](https://www.image-net.org/) data. The program is written in C++.

![image](examples/13805553_1c0c7a9fcc.jpg)
![image](examples/13805553_1c0c7a9fcc_edges.jpg)


# Dependencies
It's assumed that NPP and the cuda runtime API are installed. The code also uses the [stb library](https://github.com/nothings/stb) for reading/writing jpg files. The headers stb_image.h and stb_image_write.h are included in this codebase.

# How to compile
```
nvcc canny_npp_example.cpp -o canny_npp_example -lnppc -lnppif -lnppicc -lnppisu -lnppial -lcudart
```

# How to run
```
./canny_npp_example directory_of_input_images/ directory_of_output_images/
```

# How it all works

The program checks for the input jpgs and then for every image in the input directory:

1. Load the image from disk and load it on the GPU.
2. Convert the image to grayscale.
3. Reserve memory for scratch work (required by the canny edge detection algorithm)
4. Apply `nppiFilterCannyBorder_8u_C1R` to get the edges.
5. Copy the image of edges to the host and write to disk.



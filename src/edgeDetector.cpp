/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>


namespace fs = std::filesystem;

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

static int detectEdges(char* input, std::string output)
{
    std::ifstream infile(input, std::ifstream::in);
    if (infile.good())
    {
        std::cout << "EdgeDetector opened: <" << input << "> successfully!" << std::endl;
        infile.close();
    }
    else
    {
        std::cout << "EdgeDetector unable to open: <" << input << ">" << std::endl;
        infile.close();
        return -1;
    }

    std::string sFilename(input);
    // Declare a host image object for an 8-bit grayscale image (nppiFilterCannyBorder_8u_C1R takes 8-bit single-channel/grayscale image as input)
    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage(sFilename, oHostSrc);

    // Declare a device image using copy constructor from the host image
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // Create struct with ROI size (here we set it same as input size)
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

    // Allocate device memory for output image based on ROI
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    // Create and allocate memory for auxillary buffer required for intermediate steps of CannyBorderFilter
    int nBufferSize = 0;
    Npp8u* pScratchBufferNPP = 0;
    // Calculate scratch buffer size based destination image size
    try
    {
        NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize));
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        return -1;
    }
    catch(...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        return -1;
    }

    cudaMalloc((void**)&pScratchBufferNPP, nBufferSize);

    // Set the low and high threshold, ratio of high to threshold limit be in the ratio of (2 or 3) / 1
    // We can fine tune this parameters for each image's median pixel values.
    Npp16s nLowThreshold = 76;
    Npp16s nHighThreshold = 230;

    if (nBufferSize > 0 && pScratchBufferNPP != 0)
    {
        try
        {
            NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                        oSrcSize, oSrcOffset,
                        oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, 
                        NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
                        nLowThreshold, nHighThreshold,
                        nppiNormL2, NPP_BORDER_REPLICATE, pScratchBufferNPP));
        }
        catch (npp::Exception &rException)
        {
            std::cerr << "Program error! The following exception occurred: \n";
            std::cerr << rException << std::endl;
            std::cerr << "Aborting." << std::endl;
            return -1;
        }
        catch(...)
        {
            std::cerr << "Program error! An unknow type of exception occurred. \n";
            return -1;
        }
    }

    // Free auxillary buffer
    cudaFree(pScratchBufferNPP);

    // Create host destination object
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());

    // Copy to host destination from device 
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(output, oHostDst);

    cudaDeviceSynchronize();

    return 0;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sOuputFile;
        char *filePath;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("data/images", argv[0]);
        }

        if (fs::is_directory(filePath))
        {
            for (auto &entry : fs::directory_iterator(filePath))
            {
                if (entry.path().filename().string().rfind(".jpg"))
                {
                    std::string sOutputFile = "data/output/" + entry.path().filename().string();
                    if (-1 == detectEdges(entry.path().string().data(), sOutputFile))
                    {
                        std::cerr << "Edge detector failed !!! \n";
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        else if (fs::is_regular_file(filePath))
        {
            std::string sOutputFile = "data/output/" + fs::path(filePath).filename().string();
            if (-1 == detectEdges(filePath, sOutputFile))
            {
                std::cerr << "Edge detector failed !!! \n";
                exit(EXIT_FAILURE);
            }
        }

        exit(EXIT_SUCCESS);

    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}

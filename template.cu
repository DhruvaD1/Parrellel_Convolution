  // Histogram Equalization

  #include <wb.h>

  #define HISTOGRAM_LENGTH 256

  //@@ insert code here


  __global__ void floatusigc(float* input, unsigned char* output, int width, int height){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ro = blockIdx.y * blockDim.y + threadIdx.y;
    int i = ro *width +c+width*height *blockIdx.z;
    if(c<width && ro<height)
      output[i] = (unsigned char)(255 *input[i]);
  }

  __global__ void RGBGG(unsigned char* input, unsigned char* output, int width, int height){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ro = blockIdx.y * blockDim.y + threadIdx.y;
    int i = ro * width + c;

    if(c <width && ro <height){
      unsigned char r = input[3*i];
      unsigned char g = input[3*i +1];
      unsigned char b = input[3*i +2];
      output[i] = (unsigned char)(0.21*r+0.71*g +0.07*b);
    }
  }

  __global__ void histcc(unsigned char* input, int* output, int width, int height){
    __shared__ unsigned int histog[HISTOGRAM_LENGTH];

    int bi = threadIdx.x + threadIdx.y * blockDim.x;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ro = blockIdx.y * blockDim.y + threadIdx.y;
    int ii = ro * width + c;

    if (bi < HISTOGRAM_LENGTH){
    histog[bi] = 0;
    }
    __syncthreads();

    if(c < width && ro <  height){
      atomicAdd(&(histog[input[ii]]),1);
    }
    __syncthreads();

    if (bi < HISTOGRAM_LENGTH){
      atomicAdd(&(output[bi]),histog[bi]);
    }
  }

    __global__ void cdfcc(int* input, float* output, int image_size){
    __shared__ float histt[HISTOGRAM_LENGTH];
    int t = threadIdx.x;
    histt[t] = input[2 * blockDim.x * blockIdx.x  + t];
    histt[t + blockDim.x] = input[2 * blockDim.x * blockIdx.x + t + blockDim.x];

    int stride = 1;
    while (stride<2*HISTOGRAM_LENGTH) {
      __syncthreads();
      int index =(threadIdx.x + 1) * stride * 2 - 1;
      if ((index -stride) >= 0 && index <2 *blockDim.x){
        histt[index] += histt[index -stride];
      }
      stride = stride * 2;
    } 


    stride = blockDim.x /2;
    while (stride > 0) {
      __syncthreads();
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if ((index + stride) < 2 * blockDim.x)
        T[index + stride] += T[index];
      stride /=2;
    }


    __syncthreads();

    output[2 * (blockDim.x * blockIdx.x) + t] = histt[t] / ((float)(image_size));
    output[2 * (blockDim.x * blockIdx.x) + t + blockDim.x] = histt[t + blockDim.x] / ((float)(image_size));
  }





  __global__ void histee(unsigned char* image, float* cdf, int width, int height){
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    int ro = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = ro * width + c + width * height * blockIdx.z;

    if(c <width && ro <height){
      image[idx] = min(255.0,     max(255*(cdf[image[idx]] - cdf[0])/(1.0-cdf[0]), 0.0));
    }
  }




  __global__ void unsigcfloat(unsigned char* input, float* output, int width, int height){
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    int ro = blockIdx.y*blockDim.y + threadIdx.y;
    int i = ro * width + c + width * height * blockIdx.z;
    if(c < width && ro < height)
      output[i] = (float)(input[i]/255.0);
  }





  int main(int argc, char **argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    const char *inputImageFile;

    //@@ Insert more code here

    float *device_inputImage, *device_outputImage;
    unsigned char *device_ucharImage, *device_grayImage;
    int *device_histogram;
    float *device_cdf;
    

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    int images = imageWidth *imageHeight; //dimension

    cudaMalloc((void**)&device_inputImage,sizeof(float) * images *imageChannels);
    cudaMalloc((void**)&device_ucharImage,sizeof(unsigned char) *images * imageChannels);
    cudaMalloc((void**)&device_grayImage,sizeof(unsigned char) * images);
    cudaMalloc((void**)&device_histogram,sizeof(int) * HISTOGRAM_LENGTH);
    cudaMalloc((void**)&device_cdf,sizeof(float)*HISTOGRAM_LENGTH);
    cudaMalloc((void**)&device_outputImage,sizeof(float) * images * imageChannels);

    cudaMemcpy(device_inputImage, hostInputImageData, sizeof(float) * images * imageChannels, cudaMemcpyHostToDevice);
    cudaMemset((void*)device_histogram, 0, sizeof(int) * HISTOGRAM_LENGTH);

    dim3 blockk(32, 32, 1);
    dim3 gridd(ceil((float)imageWidth/32), ceil((float)imageHeight/32), imageChannels);
    dim3 griddy(ceil((float)imageWidth/32), ceil((float)imageHeight/32), 1);
    
    floatusigc<<<gridd, blockk>>>(device_inputImage, device_ucharImage, imageWidth, imageHeight);
    RGBGG<<<griddy, blockk>>>(device_ucharImage, device_grayImage, imageWidth, imageHeight);
    histcc<<<griddy, blockk>>>(device_grayImage, device_histogram, imageWidth, imageHeight);
    cdfcc<<<1, HISTOGRAM_LENGTH/2>>>(device_histogram, device_cdf, images);
    histee<<<gridd, blockk>>>(device_ucharImage, device_cdf, imageWidth, imageHeight);
    unsigcfloat<<<gridd, blockk>>>(device_ucharImage, device_outputImage, imageWidth, imageHeight);

    cudaMemcpy(hostOutputImageData, device_outputImage, sizeof(float) * images * imageChannels, cudaMemcpyDeviceToHost);
    




    wbSolution(args, outputImage);
    
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
  }

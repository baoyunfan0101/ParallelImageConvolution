#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define CHECK(res) if(res!=cudaSuccess){printf("cudaError = %d!\n", res);exit(-1);}
#define IMG_NUM 20

/* get time stamp */
#if __linux__
#include <sys/time.h>
#include <time.h>
double get_time(void)
{
    struct timeval tv;
    double t;

	gettimeofday(&tv, NULL);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

    return t;
}
#else
#include <windows.h>
double get_time(void)
{
    LARGE_INTEGER timer;
    static LARGE_INTEGER fre;
    static int init = 0;
    double t;

    if (init != 1) {
        QueryPerformanceFrequency(&fre);
        init = 1;
    }
    QueryPerformanceCounter(&timer);
    t = timer.QuadPart * 1. / fre.QuadPart;

    return t;
}
#endif

/* 检查设备 */
void CheckDevice()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
 
    int dev;
    for (dev = 0; dev < deviceCount; dev++) {
        int driver_version(0), runtime_version(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0)
            if (deviceProp.minor = 9999 && deviceProp.major == 9999)
                printf("\n");
        printf("Device%d:\"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driver_version);
        printf("CUDA driver version:                            %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA runtime version:                           %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
        printf("Device compute capability:                      %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
        printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
        printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
        printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
        printf("Warp size:                                      %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
        printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
        printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
    }
}

/* 文件信息头结构体 */
typedef struct MY_BITMAPFILEHEADER
{
	unsigned short bfType;        // BM字符串,bmp格式文件:0x4d42(19778)
	unsigned int   bfSize;        // 文件大小,单位:字节(2-5字节)
	unsigned short bfReserved1;   // 保留,0(6-7字节)
	unsigned short bfReserved2;   // 保留,0(8-9字节)
	unsigned int   bfOffBits;     // 从文件头到像素数据的偏移(10-13字节)
} MY_BITMAPFILEHEADER;

/* 图像信息头结构体 */
typedef struct MY_BITMAPINFOHEADER
{
	unsigned int    biSize;          // 此结构体的大小(14-17字节)
	unsigned int    biWidth;         // 图像的宽(18-21字节)
	unsigned int    biHeight;        // 图像的高(22-25字节)
	unsigned short  biPlanes;        // bmp图片的位面数,调色板的数量,1(26-27字节)
	unsigned short  biBitCount;      // 一像素所占的位数,一般为24(28-29字节)
	unsigned int    biCompression;   // 图像数据压缩的类型,不压缩:0(30-33字节)
	unsigned int    biSizeImage;     // 像素数据所占大小,bfSize-bfOffBits(34-37字节)
	unsigned int    biXPelsPerMeter; // 水平分辨率,像素/米,一般为0(38-41字节)
	unsigned int    biYPelsPerMeter; // 垂直分辨率,像素/米,一般为0(42-45字节)
	unsigned int    biClrUsed;       // 位图实际使用彩色表中的颜色索引数,使用所有调色板项:0(46-49字节)
	unsigned int    biClrImportant;  // 对图像显示有重要影响的颜色索引的数目,都重要:0(50-53字节)
} MY_BITMAPINFOHEADER;

/* 调色板 */
typedef struct RGBQuad
{
	unsigned char rgbBlue;		// 该颜色的蓝色分量,0-255
	unsigned char rgbGreen;		// 该颜色的绿色分量,0-255
	unsigned char rgbRed;		// 该颜色的红色分量,0-255
	unsigned char rgbReserved;	// 保留,0
} RGBQuad;

/* 读取特定bmp文件(4096 * 2304, 3通道) */
void read_bmp(char* filepath, unsigned char* imgData, MY_BITMAPFILEHEADER* bmf, MY_BITMAPINFOHEADER* bmi)
{
	FILE* fp;

	fp = fopen(filepath, "rb");
	if (!fp) {
		printf("failed to open %s!\n", filepath);
		exit(-1);
	}

	fread(bmf, 1, sizeof(unsigned short), fp);
	fread(&bmf->bfSize, 1, sizeof(MY_BITMAPFILEHEADER) - 4, fp);
	fread(bmi, 1, sizeof(MY_BITMAPINFOHEADER), fp);

	//fseek(fp, bmf->bfOffBits, SEEK_SET);	// 移动像素数据开始位置

	if (fread(imgData, 3 * 4096 * 2304 * sizeof(char), 1, fp) != 1) {
		fclose(fp);
		printf("%s was broken!\n", filepath);
		exit(-1);
	}

	fclose(fp);
	return;
}

/* 写入特定bmp文件(2048 * 1152, 3通道) */
void write_bmp(char* filepath, unsigned char* imgData, MY_BITMAPFILEHEADER* bmf, MY_BITMAPINFOHEADER* bmi)
{
	FILE* fp;

	fp = fopen(filepath, "wb");
	if (!fp) {
		printf("failed to open %s!\n", filepath);
		return;
	}

	fwrite(bmf, sizeof(unsigned short), 1, fp);
	fwrite(&(bmf->bfSize), sizeof(MY_BITMAPFILEHEADER) - 4, 1, fp);
	fwrite(bmi, sizeof(MY_BITMAPINFOHEADER), 1, fp);

	fwrite(imgData, 3 * 2048 * 1152 * sizeof(char), 1, fp);

	fclose(fp);
	return;
}

/* 提取B,G,R(B,G,R: (4100x2308, 1), type = unsigned char, 卷积半径2) */
__global__
void get_imgData(unsigned char* B, unsigned char* G, unsigned char* R, unsigned char* imgData)
{
	/* 当前线程提取通道 */
	register unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* 当前线程计算位置 */
	register int col = blockIdx.x * blockDim.x + threadIdx.x;
	register int row = blockIdx.y * blockDim.y + threadIdx.y;

	register int idxd = 4100 * 2308 * blockIdx.z + 4100 * row + col;	// D索引

	/* 边缘填充0 */
	if (col < 2 || col >= 4098 || row < 2 || row >= 2306) {
		D[idxd] = (unsigned char)0;
		return;
	}

	register int idxi = 3 * 4096 * 2304 * blockIdx.z + 3 * 4096 * (row - 2) + 3 * (col - 2) + threadIdx.z;	// imgData索引
	D[idxd] = imgData[idxi];

	return;
}

/* 卷积(卷积核5x5的GaussCore, 图像大小4096x2304, 展开查表) */
__global__
void conv(unsigned char* convData, unsigned char* B, unsigned char* G, unsigned char* R, double* table)
{
	/* 当前线程计算位置 */
	register int col = blockIdx.x * blockDim.x + threadIdx.x;
	register int row = blockIdx.y * blockDim.y + threadIdx.y;

	register int idxd = 4100 * 2308 * blockIdx.z + 4100 * row + col;	// D索引
	register int idxc = 3 * 4096 * 2304 * blockIdx.z + 3 * 4096 * row + 3 * col + threadIdx.z;	// convData索引

	/* 当前线程卷积通道 */
	register unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* 卷积计算 */
	convData[idxc] = (unsigned char)(
		table[D[idxd]] +
		table[256 + D[idxd + 1]] +
		table[512 + D[idxd + 2]] +
		table[256 + D[idxd + 3]] +
		table[D[idxd + 4]] +
		table[256 + D[idxd + 4100]] +
		table[768 + D[idxd + 4101]] +
		table[1024 + D[idxd + 4102]] +
		table[768 + D[idxd + 4103]] +
		table[256 + D[idxd + 4104]] +
		table[512 + D[idxd + 8200]] +
		table[1024 + D[idxd + 8201]] +
		table[1280 + D[idxd + 8202]] +
		table[1024 + D[idxd + 8203]] +
		table[512 + D[idxd + 8204]] +
		table[256 + D[idxd + 12300]] +
		table[768 + D[idxd + 12301]] +
		table[1024 + D[idxd + 12302]] +
		table[768 + D[idxd + 12303]] +
		table[256 + D[idxd + 12304]] +
		table[D[idxd + 16400]] +
		table[256 + D[idxd + 16401]] +
		table[512 + D[idxd + 16402]] +
		table[256 + D[idxd + 16403]] +
		table[D[idxd + 16404]]
		);

	return;
}

/* 池化(池化方法Max pooling, 范围2x2, 不分布) */
__global__
void Max_pooling(unsigned char* poolData, unsigned char* convData)
{
	/* 当前线程计算位置 */
	register int col = blockIdx.x * blockDim.x + threadIdx.x;
	register int row = blockIdx.y * blockDim.y + threadIdx.y;

	register int idxc = 3 * 4096 * 2304 * blockIdx.z + 2 * 3 * 4096 * row + 2 * 3 * col + threadIdx.z;	// convData索引
	register int idxp = 3 * 2048 * 1152 * blockIdx.z+ 3 * 2048 * row + 3 * col + threadIdx.z;	// poolData索引

	/* 池化计算 */
	register unsigned char tmp1 = convData[idxc], tmp2 = convData[3 + idxc],
		tmp3 = convData[3 * 4096 + idxc], tmp4 = convData[3 * 4096 + 3 + idxc];
	register unsigned char t1 = (tmp1 > tmp2) ? tmp1 : tmp2;
	register unsigned char t2 = (tmp3 > tmp4) ? tmp3 : tmp4;

	poolData[idxp] = (t1 > t2) ? t1 : t2;

	return;
}

int main()
{
	double timeuse[3] = { 0. };
	double timestart0 = get_time();	// 计时开始
	cudaError_t res;

	// 检查设备
	//CheckDevice();
	printf("Device prepared..\n");

	/* 串行读取: "png_XX.bmp" -> imgData(Host) */
	const char infile0[] = "png_";

	MY_BITMAPFILEHEADER fileHeader;	// 文件头
	MY_BITMAPINFOHEADER infoHeader;

	unsigned char* imgData;
	res = cudaMallocHost((void**)&imgData, IMG_NUM * 3 * 4096 * 2304 * sizeof(unsigned char));
	CHECK(res)

	printf("Starting to read images..\n");
	for (int i = 1; i <= IMG_NUM; ++i) {	// 循环遍历IMG_NUM张图像
		char infile[11];	// 输入文件名
		strcpy(infile, infile0);
		if (i < 10) {
			infile[4] = '0' + i;
			infile[5] = '.';
			infile[6] = 'b';
			infile[7] = 'm';
			infile[8] = 'p';
			infile[9] = '\0';
		}
		else if (i < 100) {
			infile[4] = '0' + i / 10;
			infile[5] = '0' + i - i / 10 * 10;
			infile[6] = '.';
			infile[7] = 'b';
			infile[8] = 'm';
			infile[9] = 'p';
			infile[10] = '\0';
		}

		// 读取bmp文件
		read_bmp(infile, imgData + 3 * 4096 * 2304 * (i - 1), &fileHeader, &infoHeader);
		//if (infoHeader.biBitCount != 24) {
		//	printf("%s is not 24-bit image!\n", infile);
		//	exit(-1);
		//}

		printf("\t%s read.\n", infile);
	}

	// 提前计算查表所需table
	const double littleGauss[6] = { 0.01441881 ,0.02808402,0.0350727,0.0547002,0.06831229,0.08531173 };
	double* table;
	res = cudaMallocHost((void**)&table, 6 * 256 * sizeof(double));	// 为table申请内存(1D)
	CHECK(res)
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 256; ++j) {
				table[i * 256 + j] = littleGauss[i] * j;
			}
		}

	/* 分配GPU内存: imgData(Host) -> imgData(Device) */
	printf("Allocating GPU memory..\n");

	unsigned char* convData_device;	// 存放卷积后的像素数据
	res = cudaMalloc((void**)&convData_device, IMG_NUM * 3 * 4096 * 2304 * sizeof(unsigned char));
	CHECK(res)

	unsigned char* poolData_device;	// 存放池后的像素数据
	res = cudaMalloc((void**)&poolData_device, IMG_NUM * 3 * 2048 * 1152 * sizeof(unsigned char));
	CHECK(res)

	unsigned char* imgData_device;	// 存放初始像素数据
	res = cudaMalloc((void**)&imgData_device, IMG_NUM * 3 * 4096 * 2304 * sizeof(unsigned char));
	CHECK(res)
	res = cudaMemcpy((void*)imgData_device, (void*)imgData, IMG_NUM * 3 * 4096 * 2304 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	CHECK(res)
	res = cudaFreeHost(imgData);
	CHECK(res)

	unsigned char* B_device;	// 存放B通道像素数据
	res = cudaMalloc((void**)&B_device, IMG_NUM * 4100 * 2308 * sizeof(unsigned char));
	CHECK(res)

	unsigned char* G_device;	// 存放G通道像素数据
	res = cudaMalloc((void**)&G_device, IMG_NUM * 4100 * 2308 * sizeof(unsigned char));
	CHECK(res)

	unsigned char* R_device;	// 存放R通道像素数据
	res = cudaMalloc((void**)&R_device, IMG_NUM * 4100 * 2308 * sizeof(unsigned char));
	CHECK(res)

	double* table_device;	// 存放table
	res = cudaMalloc((void**)&table_device, 6 * 256 * sizeof(double));
	CHECK(res)
	res = cudaMemcpy((void*)table_device, (void*)table, 6 * 256 * sizeof(double), cudaMemcpyHostToDevice);
	CHECK(res)
	res = cudaFreeHost(table);
	CHECK(res)

	/* 并行预处理: imgData(Device) -> B,G,R(Device) */
	printf("Proprocessing parallelly..\n");
	double timestart = get_time();	// 计时开始

	dim3 grid0(205, 577, IMG_NUM), block0(20, 4, 3);	//dim3: blockIdx -> grid(\), threadIdx -> block(blockDim)
	get_imgData<<<grid0, block0>>>(B_device, G_device, R_device, imgData_device);

	cudaDeviceSynchronize();	// 线程同步
	double timeend = get_time();	// 计时结束
	timeuse[0] = timeend - timestart;

	res = cudaFree(imgData_device);
	CHECK(res)

	/* 并行卷积: B,G,R(Device) -> convData(Device) */
	printf("Convoluting parallelly..\n");
	timestart = get_time();	// 计时开始

	dim3 grid1(256, 288, IMG_NUM), block1(16, 8, 3);	//dim3: blockIdx -> grid(\), threadIdx -> block(blockDim)
	conv<<<grid1, block1>>>(convData_device, B_device, G_device, R_device, table_device);

	cudaDeviceSynchronize();	// 线程同步
	timeend = get_time();	// 计时结束
	timeuse[1] = timeend - timestart;

	res = cudaFree(table_device);
	CHECK(res)
	res = cudaFree(R_device);
	CHECK(res)
	res = cudaFree(G_device);
	CHECK(res)
	res = cudaFree(B_device);
	CHECK(res)

	/* 并行池化: convData(Device) -> poolData(Device) */
	printf("Pooling parallelly..\n");
	timestart = get_time();	// 计时开始

	dim3 grid2(256, 288, IMG_NUM), block2(8, 4, 3);	//dim3: blockIdx -> grid(\), threadIdx -> block(blockDim)
	Max_pooling<<<grid2, block2>>>(poolData_device, convData_device);

	cudaDeviceSynchronize();	// 线程同步
	timeend = get_time();	// 计时结束
	timeuse[2] = timeend - timestart;

	res = cudaFree(convData_device);
	CHECK(res)

	/* poolData(Device) -> poolData(Host) */
	unsigned char* poolData;
	res = cudaMallocHost((void**)&poolData, IMG_NUM * 3 * 2048 * 1152 * sizeof(unsigned char));
	CHECK(res)
	res = cudaMemcpy((void*)poolData, (void*)poolData_device, IMG_NUM * 3 * 2048 * 1152 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	CHECK(res)

	res = cudaFree(poolData_device);
	CHECK(res)

	// 池化后,修改文件头
	fileHeader.bfSize = 7077942;
	infoHeader.biWidth = 2048;
	infoHeader.biHeight = 1152;

	/* 串行写入: poolData(Host) -> "CUDA_XX.bmp" */
	const char outfile0[] = "CUDA_";

	printf("Starting to write images..\n");
	for (int i = 1; i <= IMG_NUM; ++i) {	// 循环遍历IMG_NUM张图像
		char outfile[29];	// 输出文件名
		strcpy(outfile, outfile0);
		if (i < 10) {
			outfile[22] = '0' + i;
			outfile[23] = '.';
			outfile[24] = 'b';
			outfile[25] = 'm';
			outfile[26] = 'p';
			outfile[27] = '\0';
		}
		else if (i < 100) {
			outfile[22] = '0' + i / 10;
			outfile[23] = '0' + i - i / 10 * 10;
			outfile[24] = '.';
			outfile[25] = 'b';
			outfile[26] = 'm';
			outfile[27] = 'p';
			outfile[28] = '\0';
		}

		// 写入bmp文件
		write_bmp(outfile, poolData + 3 * 2048 * 1152 * (i - 1), &fileHeader, &infoHeader);

		printf("\t%s is written.\n", outfile);
	}

	res = cudaFreeHost(poolData);
	CHECK(res)

	double timeend0 = get_time();	// 计时结束
	printf("\nGPU preprocessing time: %lfs\nGPU convolution time: %lfs\nGPU pooling time: %lfs\ntotal runtime: %lfs\n", timeuse[0], timeuse[1], timeuse[2], timeend0 - timestart0);

	return 0;
}
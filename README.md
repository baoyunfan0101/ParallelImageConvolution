# Parallel Image Convolution

## 测试环境
ubuntu 20.04  
MPI / Pthread / OpenMP / CUDA

## 设计思路
### BMP图像的读写
可以考虑使用OpenCV实现，但自己编写也并不困难。  

首先，bmp文件包含四个部分，分别为文件信息头、图像信息头、调色板和像素数据。其中前三部分可以用下面结构定义：
```
/* 文件信息头结构体 */
typedef struct tagBITMAPFILEHEADER
{
	unsigned short bfType;        // BM字符串, bmp格式文件:0x4d42(19778)
	unsigned int   bfSize;        // 文件大小,单位:字节(2-5字节)
	unsigned short bfReserved1;   // 保留,0(6-7字节)
	unsigned short bfReserved2;   // 保留,0(8-9字节)
	unsigned int   bfOffBits;     // 从文件头到像素数据的偏移(10-13字节)
} BITMAPFILEHEADER;

/* 图像信息头结构体 */
typedef struct tagBITMAPINFOHEADER
{
	unsigned int    biSize;          // 此结构体的大小(14-17字节)
	long            biWidth;         // 图像的宽(18-21字节)
	long            biHeight;        // 图像的高(22-25字节)
	unsigned short  biPlanes;        // bmp图片的位面数,调色板的数量,1(26-27字节)
	unsigned short  biBitCount;      // 一像素所占的位数,一般为24(28-29字节)
	unsigned int    biCompression;   // 图象数据压缩的类型,不压缩:0(30-33字节)
	unsigned int    biSizeImage;     // 像素数据所占大小,bfSize-bfOffBits(34-37字节)
	long            biXPelsPerMeter; // 水平分辨率,像素/米,一般为0(38-41字节)
	long            biYPelsPerMeter; // 垂直分辨率,像素/米,一般为0(42-45字节)
	unsigned int    biClrUsed;       // 位图实际使用彩色表中的颜色索引数,使用所有调色板项:0(46-49字节)
	unsigned int    biClrImportant;  // 对图象显示有重要影响的颜色索引的数目,都重要:0(50-53字节)
} BITMAPINFOHEADER;

/* 调色板 */
typedef struct RGBQuad
{
	unsigned char rgbBlue;		// 该颜色的蓝色分量,0-255
	unsigned char rgbGreen;	// 该颜色的绿色分量,0-255
	unsigned char rgbRed;		// 该颜色的红色分量,0-255
	unsigned char rgbReserved;	// 保留,0
} RGBQuad;
```

仅考虑24位位图，其像素数据分为蓝、绿、红三个通道，每个通道占8位。完整提取出卷积所需的像素数据后，可以存入B、G、R三个数组中以便后续操作。其中读取文件的函数如下：
```
/* 读取bmp文件,返回像素数据 */
char* Read_bmp(char* filepath, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi)
{
	unsigned char* imgData;
	FILE* fp;

	fp = fopen(filepath, "rb");
	if (!fp) {
		printf("bmp文件打开失败！\n");
		return NULL;
	}

	fread(bmf, 1, sizeof(unsigned short), fp);
	fread(&bmf->bfSize, 1, sizeof(BITMAPFILEHEADER) - 4, fp);
	//showBmpHead(*bmf);
	fread(bmi, 1, sizeof(BITMAPINFOHEADER), fp);
	//showBmpInfoHead(*bmi);

	int width = bmi->biWidth;
	int height = bmi->biHeight;
	int bitCount = bmi->biBitCount;
	imgData = (unsigned char*)malloc((bitCount / (8 * sizeof(char))) * width * height * sizeof(char));
	if (!imgData) {
		printf("内存申请失败！\n");
		return NULL;
	}
	fseek(fp, bmf->bfOffBits, SEEK_SET);	// 移动像素数据开始位置

	if (fread(imgData, (bitCount / (8 * sizeof(char))) * width * height * sizeof(char), 1, fp) != 1) {
		free(imgData);
		fclose(fp);
		printf("bmp文件损坏！\n");
		return NULL;
	}

	fclose(fp);
	return imgData;
}
```

将卷积后的数据重新写入新的bmp文件中时，需要注意的是，24位位图不需要调色板。其函数如下：
```
/* 写入bmp文件 */
void Write_bmp(char* filepath, unsigned char* imgData, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi)
{
	FILE* fp;
	long height = bmi->biHeight;
	unsigned int dwLineBytes = (bmi->biBitCount / (8 * sizeof(char))) * bmi->biWidth;
	fp = fopen(filepath, "wb");
	if (!fp) {
		printf("bmp文件写入失败！\n");
		return;
	}

	fwrite(bmf, sizeof(unsigned short), 1, fp);
	fwrite(&(bmf->bfSize), sizeof(BITMAPFILEHEADER) - 4, 1, fp);
	fwrite(bmi, sizeof(BITMAPINFOHEADER), 1, fp);

	/* 24位真彩色图像，无调色板 */

	fwrite(imgData, dwLineBytes * height, 1, fp);
}
```

### MPI
先将图像按照像素分成若干部分（数量等同于进程数），通过MPI_Scatterv函数分发数据，每个进程计算自己对应部分的B、G、R的卷积，计算完成后通过MPI_Gatherv函数集聚到父进程。

### Pthread
首先，定义如下的结构体mythread，用于向新线程传递其所需的所有信息。
```
/* 线程创建结构体 */
typedef struct mythread {
	int myid;	// 当前进程编号
	unsigned int** B;	// 待卷积数据基址
	unsigned int** G;
	unsigned int** R;
	unsigned char* convData;	// 结果存放基址
	double* table;	// 查表基址
}mythread;
```

在并行部分中，为便于修改，预先定义线程总数为宏NUMPROCS。

首先为存放结果的数组convData申请空间，由于各个线程间共用内存资源，因此所有线程的计算结果将直接放入convData的对应位置中。在每次循环中，利用pthread_create函数创建新线程，并将新线程所需的数据存入结构体中，传递给新线程。在并行部分结束时，利用pthread_join函数等待所有线程终止，再输出结果即可。
```
/* 并行部分 */
unsigned char* convData = (unsigned char*)malloc((infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight * sizeof(char));
if (!convData) {
	printf("内存申请失败！\n");
	return 0;
}

pthread_t* pt = (pthread_t*)malloc(NUMPROCS * sizeof(pthread_t));

for (int myid = 0; myid < NUMPROCS; ++myid) {	// myid - 当前进程编号
	mythread* pmythread = (mythread*)malloc(sizeof(mythread));
	if (!pmythread) {
		printf("内存申请失败！\n");
		return 0;
	}
	pmythread->myid = myid;
	pmythread->B = B;
	pmythread->G = G;
	pmythread->R = R;
	pmythread->convData = convData;
	pmythread->table = table;

	int error = pthread_create(pt + myid, NULL, mythreadfun, (void*)pmythread);
	if (error) {
		printf("%d#线程创建失败！\n", myid);
		return 0;
	}
}

for (int myid = 0; myid < NUMPROCS; ++myid) {
	void* p = NULL;
	int error = pthread_join(*(pt + myid), p);
	if (error) {
		printf("%d#线程未终止！\n");
		return 0;
	}
}
```

### OpenMP
由于OpenMP能够简便地自动对程序进行并行处理，因此并行部分的设计较为简单，在用于卷积计算的for循环前插入pragma语句，使循环并行执行即可。

特别地，由于所有线程将共用内存资源，且某次循环在某个线程中执行是随机的，因此应采用private子句指定合适的变量（如循环变量i、j等），使每个线程都拥有该变量的私有副本，防止循环发生混乱。这里使用的pragma语句如下：
```
#pragma omp parallel for num_threads(1) private(i,j)
```

### CUDA
在CUDA并行程序中，可以通过`__global__ void kernel()`定义由GPU中的线程执行的kernel函数，通过主函数中的`kernel<<<grid, block>>>()`语句调用kernel函数。其中，grid和block为dim3类型的变量。每个kernel函数将启动GPU中的一个网格（grid），grid和block分别指定一个网格中的块（block）维度和一个块中的线程（thread）维度。

根据以上分析，我们可以让gridDim.x和blockDim.x对应图片的x轴（即要求gridDim.x * blockDim.x为图片的宽），让gridDim.y和blockDim.y对应图片的y轴（即要求gridDim.y * blockDim.y为图片的高），由此分别能够通过`blockIdx.x * blockDim.x + threadIdx.x`和`blockIdx.y * blockDim.y + threadIdx.y`计算得到图片的x轴、y轴坐标。再让gridDim.z对应图片的序号（即要求gridDim.z = 20），让blockDim.z对应通道的序号（即要求blockDim.z = 3），在每个线程中便能够进一步定位在一张图片中的一个像素的一个通道上，如下图所示。

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/CUDA.png)

在kernel函数中，先求得当前线程需要计算的位置，后并行计算并将结果写入内存指定位置上；在主函数中，则先分配足够的GPU内存，将并行计算所需的数据从Host（CPU内存）拷贝至Device（GPU内存）中，调用kernel函数，再视情况决定是否将Device中的结果拷贝至Host中。

## 优化方法
### 卷积计算的优化
这里实现算法的卷积核是5*5的高斯核，因此以此为例进行说明，代码则采用CUDA编程。

在并行卷积的计算上，每个线程最基本的方法是采用双重循环乘法，即将25次乘法的结果求和得到卷积后的结果。其中，为减少访存的次数，各通道的数据及高斯核均采用一维存储。
```
/* kernel函数1: 双重循环乘法 */
__global__
void conv1(unsigned char* convData, unsigned char* B, unsigned char* G, unsigned char* R, double* GaussCore_1D)
{
	/* 当前线程计算位置 */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int idxc = blockIdx.z * 3 * 4096 * 2304 + 3 * 4096 * row + 3 * col + threadIdx.z;	// convData索引
	register int idxd = blockIdx.z * 4100 * 2308 + 4100 * row + col;	// D索引

	/* 当前线程卷积通道 */
	unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* 卷积计算 */
	register double tmp = 0;

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			tmp += D[idxd + 4100 * i + j] * GaussCore_1D[5 * i + j];
		}
	}

	convData[idxc] = (unsigned char)(tmp);
}
```

双重循环乘法经过优化可以得到展开乘法，即不采用循环，而是将25次乘法的语句全部展开。这种方法不但可以减少循环所消耗的资源，也便于直接给出高斯核的数据，无需再为高斯核申请内存和访问内存。下面的代码有省略。
```
/* kernel函数2: 展开乘法 */
__global__
void conv2(unsigned char* convData, unsigned char* B, unsigned char* G, unsigned char* R, double* GaussCore_1D)
{
	/* 当前线程计算位置 */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int idxc = blockIdx.z * 3 * 4096 * 2304 + 3 * 4096 * row + 3 * col + threadIdx.z;	// convData索引
	register int idxd = blockIdx.z * 4100 * 2308 + 4100 * row + col;	// D索引

	/* 当前线程卷积通道 */
	unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* 卷积计算 */
	convData[idxc] = (unsigned char)(
		D[idxd] * 0.01441881 +
		…… +
		D[idxd + 16404] * 0.01441881
		);
}
```

对于一个具体问题，其卷积核是确定的，因此可采用查表法优化卷积过程。

这里，依据高斯核GaussCore的对称性，不难看出其中只有六个不同数值，不妨以下图左侧红色三角形中的六个数值为例。将这六个值视为1×6的一维向量，则每个值与0至255共256个数的乘积可以写入如下图右侧所示的6×256的矩阵table中。

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/GaussCore.png)

提前将高斯核中的6个不同的浮点数与每个通道可能出现的256个整数的乘法结果求出，在卷积过程中以查表替代乘法计算。经测试，程序的效率提升显著。
```
/* kernel函数3: 展开查表 */
__global__
void conv3(unsigned char* convData, unsigned char* B, unsigned char* G, unsigned char* R, double* table)
{
	/* 当前线程计算位置 */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int idxc = blockIdx.z * 3 * 4096 * 2304 + 3 * 4096 * row + 3 * col + threadIdx.z;	// convData索引
	register int idxd = blockIdx.z * 4100 * 2308 + 4100 * row + col;	// D索引

	/* 当前线程卷积通道 */
	unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* 卷积计算 */
	convData[idxc] = (unsigned char)(
		table[D[idxd]] +
		…… +
		table[D[idxd + 16404]]
		);
}
```

### 并行规约
在实际应用中，卷积过程之后通常要对图像进行池化。

这里以2*2的最大值池化为例，可以采用二叉树算法对其进行并行规约。池化过程可以视为每4个像素的每个通道先两两求出最大值t1和t2，再从t1和t2中求出最大值即最终结果res。

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/Pooling.png)

若池化的范围较大，这种优化的方法能够发挥出更明显的作用。

### 内存分配的优化
当使用CUDA时，在内存分配上有了更多的优化空间。

分配CPU可以采用普通C语言的malloc函数。但若该CPU内存需要实现Host（CPU内存）与Device（GPU内存）的通信，则更好的选择是采用Cuda提供的cudaMallocHost函数。它与malloc的区别是可以将CPU内存与GPU内存分配成相同的格式，以大幅度提高cudaMemcpy函数的效率。与之配套的内存释放函数是cudaFree。

同时，在处理图片数据的过程中，一个线程访问的数据可能具有一定的空间局部性，即在位置上接近但在地址上并不连续，如下图所示。

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/TextureCache.png)

这种情况非常适用纹理内存。纹理内存在使用时，首先先通过`texture<TYPE>`将输入的数据声明为texture类型的引用，再通过cudaBindTexture函数为内存绑定纹理。在访问纹理内存时，需要根据纹理选择相应的拾取函数（如tex1Dfetch函数），解绑纹理可以采用cudaUnbindTexture函数。

另外，在编写CUDA并行程序时，应尽可能减少Host（CPU内存）与Device（GPU内存）间的内存拷贝。特别是在数据体量较大时，内存拷贝所消耗的资源不容小觑。

### 代码细节的优化
在卷积进行到图像边缘时，会进行特殊的处理。这里可以直接依据卷积核的边长为图像的边缘填充0，以节约卷积计算时的判断次数。

在编码过程中，对于多次计算的表达示，应将其定义为临时变量，以减少重复计算；对于多次引用的变量，应将其定义为寄存器变量，以减少访问内存的次数。

## 效果展示
实验的原始图像和卷积后的图像分别如下所示。

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/data1.png)

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/res1.png)

将上面的两张图像部分放大后如下面两图所示。通过放大后的图像可以明显看出，图像高斯模糊效果明显。

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/data2.png)

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/res2.png)

## 参考文献
[1] Peter S. Pacheco. 并行程序设计导论[M]. 北京: 机械工业出版社, 2012.

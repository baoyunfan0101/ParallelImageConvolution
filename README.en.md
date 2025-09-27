# Parallel Image Convolution
<div align="right">
	[<a href="README.md">中文</a> | English(Current)</a>]
</div>

## Test Environment
ubuntu 20.04  
MPI / Pthread / OpenMP / CUDA

## Design Ideas
### Reading and Writing BMP Images
We could consider using `OpenCV`, but writing our own implementation is not very difficult.

First, a bmp file consists of four parts: file header, image header, color palette, and pixel data. The first three can be defined with the following structures:

```cpp
/* File header structure */
typedef struct tagBITMAPFILEHEADER
{
	unsigned short bfType;        // BM string, bmp format file: 0x4d42 (19778)
	unsigned int   bfSize;        // File size, in bytes (2-5 bytes)
	unsigned short bfReserved1;   // Reserved, 0 (6-7 bytes)
	unsigned short bfReserved2;   // Reserved, 0 (8-9 bytes)
	unsigned int   bfOffBits;     // Offset from file header to pixel data (10-13 bytes)
} BITMAPFILEHEADER;

/* Image header structure */
typedef struct tagBITMAPINFOHEADER
{
	unsigned int    biSize;          // Size of this structure (14-17 bytes)
	long            biWidth;         // Image width (18-21 bytes)
	long            biHeight;        // Image height (22-25 bytes)
	unsigned short  biPlanes;        // Number of planes, usually 1 (26-27 bytes)
	unsigned short  biBitCount;      // Bits per pixel, typically 24 (28-29 bytes)
	unsigned int    biCompression;   // Compression type, 0 = no compression (30-33 bytes)
	unsigned int    biSizeImage;     // Size of pixel data, bfSize - bfOffBits (34-37 bytes)
	long            biXPelsPerMeter; // Horizontal resolution, pixels/meter, usually 0 (38-41 bytes)
	long            biYPelsPerMeter; // Vertical resolution, pixels/meter, usually 0 (42-45 bytes)
	unsigned int    biClrUsed;       // Number of colors used, 0 = all (46-49 bytes)
	unsigned int    biClrImportant;  // Number of important colors, 0 = all (50-53 bytes)
} BITMAPINFOHEADER;

/* Color palette */
typedef struct RGBQuad
{
	unsigned char rgbBlue;		// Blue component, 0-255
	unsigned char rgbGreen;		// Green component, 0-255
	unsigned char rgbRed;		// Red component, 0-255
	unsigned char rgbReserved;	// Reserved, 0
} RGBQuad;
```

We only consider 24-bit bitmaps here. Pixel data is divided into blue, green, and red channels, each occupying 8 bits. Once the pixel data needed for convolution is extracted, it can be stored in arrays B, G, and R for later operations. The function for reading files is as follows:

```cpp
/* Read bmp file, return pixel data */
char* Read_bmp(char* filepath, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi)
{
	unsigned char* imgData;
	FILE* fp;

	fp = fopen(filepath, "rb");
	if (!fp) {
		printf("Failed to open bmp file!\n");
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
		printf("Memory allocation failed!\n");
		return NULL;
	}
	fseek(fp, bmf->bfOffBits, SEEK_SET);	// Move to start of pixel data

	if (fread(imgData, (bitCount / (8 * sizeof(char))) * width * height * sizeof(char), 1, fp) != 1) {
		free(imgData);
		fclose(fp);
		printf("bmp file is corrupted!\n");
		return NULL;
	}

	fclose(fp);
	return imgData;
}
```

When writing convolution results back into a new bmp file, note that a 24-bit bitmap does not need a color palette. The function is as follows:

```cpp
/* Write bmp file */
void Write_bmp(char* filepath, unsigned char* imgData, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi)
{
	FILE* fp;
	long height = bmi->biHeight;
	unsigned int dwLineBytes = (bmi->biBitCount / (8 * sizeof(char))) * bmi->biWidth;
	fp = fopen(filepath, "wb");
	if (!fp) {
		printf("Failed to write bmp file!\n");
		return;
	}

	fwrite(bmf, sizeof(unsigned short), 1, fp);
	fwrite(&(bmf->bfSize), sizeof(BITMAPFILEHEADER) - 4, 1, fp);
	fwrite(bmi, sizeof(BITMAPINFOHEADER), 1, fp);

	/* 24-bit true color image, no palette */

	fwrite(imgData, dwLineBytes * height, 1, fp);
}
```

### MPI
First, divide the image into several parts based on pixels (equal to the number of processes). Use `MPI_Scatterv` to distribute the data. Each process performs convolution on its corresponding B, G, R sections. After completion, results are gathered back to the parent process with `MPI_Gatherv`.

### Pthread
First, define the following structure `mythread` to pass all required data to new threads:

```cpp
/* Thread creation structure */
typedef struct mythread {
	int myid;	// Current process ID
	unsigned int** B;	// Data to be convolved
	unsigned int** G;
	unsigned int** R;
	unsigned char* convData;	// Base address for results
	double* table;	// Lookup table
}mythread;
```

In the parallel section, for easier modification, the total number of threads is predefined as macro `NUMPROCS`.

Allocate memory for result array `convData`. Since threads share memory, all results are stored directly in their corresponding positions. In each loop, create new threads with `pthread_create`, passing required data through the structure. At the end, use `pthread_join` to wait for threads to terminate before outputting results.

```cpp
/* Parallel section */
unsigned char* convData = (unsigned char*)malloc((infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight * sizeof(char));
if (!convData) {
	printf("Memory allocation failed!\n");
	return 0;
}

pthread_t* pt = (pthread_t*)malloc(NUMPROCS * sizeof(pthread_t));

for (int myid = 0; myid < NUMPROCS; ++myid) {	// myid - current process ID
	mythread* pmythread = (mythread*)malloc(sizeof(mythread));
	if (!pmythread) {
		printf("Memory allocation failed!\n");
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
		printf("%d# Thread creation failed!\n", myid);
		return 0;
	}
}

for (int myid = 0; myid < NUMPROCS; ++myid) {
	void* p = NULL;
	int error = pthread_join(*(pt + myid), p);
	if (error) {
		printf("%d# Thread did not terminate!\n", myid);
		return 0;
	}
}
```

### OpenMP
Since `OpenMP` can automatically parallelize code easily, the parallel design here is simple. Insert a `pragma` statement before the convolution loop to make it execute in parallel.

Specifically, since all threads share memory and loop execution order is random, use the `private` clause to assign private copies of variables (e.g., i, j) to each thread, preventing confusion. Example:

```cpp
#pragma omp parallel for num_threads(1) private(i,j)
```

### CUDA
In CUDA programs, define a `kernel` function with `__global__ void kernel()` executed by GPU threads. Call it in the main function with `kernel<<<grid, block>>>()`. Here, `grid` and `block` are `dim3` variables specifying grid and block dimensions.

For mapping:
* `gridDim.x * blockDim.x = image width`
* `gridDim.y * blockDim.y = image height`

Coordinates:
* x = `blockIdx.x * blockDim.x + threadIdx.x`
* y = `blockIdx.y * blockDim.y + threadIdx.y`

`gridDim.z` = number of images (e.g., 20), `blockDim.z` = number of channels (3). Each thread thus processes one pixel channel:

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/CUDA.png)

Inside `kernel`, compute the target position, calculate convolution in parallel, and store results. In main, allocate GPU memory, copy data from host (CPU) to device (GPU), launch `kernel`, and optionally copy results back.

## Optimization Methods
### Convolution Optimization
Here, the convolution core is a 5×5 Gaussian core. Using CUDA as an example:

The basic approach: nested loops to compute 25 multiplications and sums. Data is stored in 1D arrays to reduce memory access.

```cpp
/* Kernel function 1: Nested loops */
__global__
void conv1(unsigned char* convData, unsigned char* B, unsigned char* G, unsigned char* R, double* GaussCore_1D)
{
	/* Current thread position */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int idxc = blockIdx.z * 3 * 4096 * 2304 + 3 * 4096 * row + 3 * col + threadIdx.z;	// convData index
	register int idxd = blockIdx.z * 4100 * 2308 + 4100 * row + col;	// D index

	/* Current channel */
	unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* Convolution */
	register double tmp = 0;

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			tmp += D[idxd + 4100 * i + j] * GaussCore_1D[5 * i + j];
		}
	}

	convData[idxc] = (unsigned char)(tmp);
}
```

Optimized: loop unrolling — expand 25 multiplications directly, reducing loop overhead and avoiding Gaussian core memory access:

```cpp
/* Kernel function 2: Unrolled multiplications */
__global__
void conv2(unsigned char* convData, unsigned char* B, unsigned char* G, unsigned char* R, double* GaussCore_1D)
{
	/* Current thread position */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int idxc = blockIdx.z * 3 * 4096 * 2304 + 3 * 4096 * row + 3 * col + threadIdx.z;	// convData index
	register int idxd = blockIdx.z * 4100 * 2308 + 4100 * row + col;	// D index

	/* Current channel */
	unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* Convolution */
	convData[idxc] = (unsigned char)(
		D[idxd] * 0.01441881 +
		…… +
		D[idxd + 16404] * 0.01441881
		);
}
```

Further optimization: lookup table. Gaussian core has only 6 distinct values (due to symmetry). Precompute their products with 0–255 and store in a 6×256 table. During convolution, replace multiplications with lookups:

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/GaussCore.png)

```cpp
/* Kernel function 3: Lookup table */
__global__
void conv3(unsigned char* convData, unsigned char* B, unsigned char* G, unsigned char* R, double* table)
{
	/* Current thread position */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int idxc = blockIdx.z * 3 * 4096 * 2304 + 3 * 4096 * row + 3 * col + threadIdx.z;	// convData index
	register int idxd = blockIdx.z * 4100 * 2308 + 4100 * row + col;	// D index

	/* Current channel */
	unsigned char* D = (threadIdx.z == 0) ? B : ((threadIdx.z == 1) ? G : R);

	/* Convolution */
	convData[idxc] = (unsigned char)(
		table[D[idxd]] +
		…… +
		table[D[idxd + 16404]]
		);
}
```

### Parallel Reduction
In real applications, pooling often follows convolution.

For example, 2×2 max pooling: use binary tree reduction. Pooling = compute t1, t2 as pairwise max of 4 pixels, then result = max(t1, t2):

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/Pooling.png)

This method is more beneficial for larger pooling windows.

### Memory Allocation Optimization
With CUDA, memory allocation can be further optimized:
* CPU memory: `malloc` is fine.
* For CPU–GPU communication: prefer `cudaMallocHost`, which allocates pinned memory compatible with GPU, improving `cudaMemcpy` performance. Free with `cudaFree`.

Texture memory is useful for spatial locality (nearby but not contiguous addresses):

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/TextureCache.png)

Declare input as `texture<TYPE>`, bind with `cudaBindTexture`, fetch with functions like `tex1Dfetch`, unbind with `cudaUnbindTexture`.

Also, minimize host–device memory transfers, especially for large data.

### Code-Level Optimizations
At image edges during convolution, directly pad with zeros to avoid extra checks.

For repeated expressions, use temporary variables to reduce recomputation. For frequently used variables, declare as `register` to reduce memory access.

## Results
The original and convolved images:

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/data1.png)

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/res1.png)

Zoomed-in comparison (Gaussian blur effect is obvious):

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/data2.png)

![image](https://github.com/baoyunfan0101/ParallelImageConvolution/blob/main/static/res2.png)

## References
[1] Peter S. Pacheco. An Introduction to Parallel Programming [M]. Beijing: China Machine Press, 2012.

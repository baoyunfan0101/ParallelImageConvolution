#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <malloc.h>
#include "omp.h"

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
	unsigned int    biWidth;         // 图像的宽(18-21字节)
	unsigned int    biHeight;        // 图像的高(22-25字节)
	unsigned short  biPlanes;        // bmp图片的位面数,调色板的数量,1(26-27字节)
	unsigned short  biBitCount;      // 一像素所占的位数,一般为24(28-29字节)
	unsigned int    biCompression;   // 图象数据压缩的类型,不压缩:0(30-33字节)
	unsigned int    biSizeImage;     // 像素数据所占大小,bfSize-bfOffBits(34-37字节)
	unsigned int    biXPelsPerMeter; // 水平分辨率,像素/米,一般为0(38-41字节)
	unsigned int    biYPelsPerMeter; // 垂直分辨率,像素/米,一般为0(42-45字节)
	unsigned int    biClrUsed;       // 位图实际使用彩色表中的颜色索引数,使用所有调色板项:0(46-49字节)
	unsigned int    biClrImportant;  // 对图象显示有重要影响的颜色索引的数目,都重要:0(50-53字节)
} BITMAPINFOHEADER;

/* 调色板 */
typedef struct RGBQuad
{
	unsigned char rgbBlue;		// 该颜色的蓝色分量,0-255
	unsigned char rgbGreen;		// 该颜色的绿色分量,0-255
	unsigned char rgbRed;		// 该颜色的红色分量,0-255
	unsigned char rgbReserved;	// 保留,0
} RGBQuad;

/* 展示BITMAPFILEHEADER,调试用 */
void showBmpHead(BITMAPFILEHEADER bmf)
{
	printf("bfSize: %dkb\n", bmf.bfSize / 1024);
	printf("bfOffBits: %d\n", bmf.bfOffBits);
}

/* 展示BITMAPINFOHEADER,调试用 */
void showBmpInfoHead(BITMAPINFOHEADER bmi)
{
	printf("bfSize: %d\n", bmi.biSize);
	printf("biWidth: %d\n", bmi.biWidth);
	printf("biHeight: %d\n", bmi.biHeight);
	printf("biPlanes: %d\n", bmi.biPlanes);
	printf("biBitCount: %d\n", bmi.biBitCount);
}

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

	/* 24位真彩色图像，无调色板
	RGBQuad* IpRGBQuad = (RGBQuad*)malloc(1 * sizeof(RGBQuad));
	IpRGBQuad[0].rgbRed = 0;
	IpRGBQuad[0].rgbGreen = 0;
	IpRGBQuad[0].rgbBlue = 0;
	IpRGBQuad[0].rgbReserved = 0;
	fwrite(IpRGBQuad, sizeof(RGBQuad), 1, fp); */

	fwrite(imgData, dwLineBytes * height, 1, fp);
}

/* 提取像素数据(B,G,R分别计算) */
void Get_imgData(unsigned int** B, unsigned int** G, unsigned int** R,
	unsigned char* imgData, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi,
	int convR)	// convR为卷积半径
{
	int h = bmi->biHeight + 2 * convR;
	int w = bmi->biWidth + 2 * convR;

	unsigned int dwLineBytes = (bmi->biBitCount / (8 * sizeof(char))) * bmi->biWidth;
	for (int i = 0; i < h; ++i) {
		if (i < convR || i >= h - convR)
			for (int j = 0; j < w; ++j) {
				B[i][j] = 0;
				G[i][j] = 0;
				R[i][j] = 0;
			}
		else {
			register int x = i - convR;
			for (int j = 0; j < w; ++j) {
				if (j < convR || j >= w - convR) {
					for (int k = 0; k < h; ++k) {
						B[k][j] = 0;
						G[k][j] = 0;
						R[k][j] = 0;
					}
				}
				else {
					int y = j - convR;
					register int tmp = x * dwLineBytes + y * 3;
					B[i][j] = (unsigned int)(*(imgData + tmp + 0));
					G[i][j] = (unsigned int)(*(imgData + tmp + 1));
					R[i][j] = (unsigned int)(*(imgData + tmp + 2));
				}
			}
		}
	}
}

/* 展示像素矩阵,并写入文件 */
void Show_res(int** a, char* filepath, int h, int w)
{
	FILE* fp;
	fp = fopen(filepath, "wb");
	if (!fp) {
		printf("结果写入失败！\n");
		return;
	}

	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			printf("%d ", a[i][j]);
			fprintf(fp, "%d ", a[i][j]);
		}
		printf("\n");
		fprintf(fp, "\n");
	}
	fclose(fp);
}

/* 高斯核 */
double GaussCore[5][5] = {
	{0.01441881, 0.02808402, 0.0350727, 0.02808402, 0.01441881},
	{0.02808402, 0.0547002, 0.06831229, 0.0547002, 0.02808402},
	{0.0350727, 0.06831229, 0.08531173, 0.06831229, 0.0350727},
	{0.02808402, 0.0547002, 0.06831229, 0.0547002, 0.02808402},
	{0.01441881, 0.02808402, 0.0350727, 0.02808402, 0.01441881}
};

/* 主函数 */
int main(int argc, char** argv)
{
	/* 读取bmp文件 */
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;
	unsigned char* imgData = Read_bmp("data.bmp", &fileHeader, &infoHeader);
	if (!imgData)
		return 0;
	else if (infoHeader.biBitCount != 24) {
		printf("暂不支持非真彩色图像！\n");
		return 0;
	}

	/* 提取像素数据 */
	int convR = 2;	// 卷积核半径
	int h = infoHeader.biHeight + 2 * convR;
	int w = infoHeader.biWidth + 2 * convR;

	/* 将B,G,R提取为二维数组 */
	unsigned int** B = (unsigned int**)malloc(h * sizeof(unsigned int*));	// 为B,G,R申请空间(二维数组)
	unsigned int** G = (unsigned int**)malloc(h * sizeof(unsigned int*));
	unsigned int** R = (unsigned int**)malloc(h * sizeof(unsigned int*));
	if (!B || !G || !R) {
		printf("内存申请失败！\n");
		return 0;
	}
	for (int i = 0; i < h; ++i)
		B[i] = (unsigned int*)malloc(w * sizeof(unsigned int));
	for (int i = 0; i < h; ++i)
		G[i] = (unsigned int*)malloc(w * sizeof(unsigned int));
	for (int i = 0; i < h; ++i)
		R[i] = (unsigned int*)malloc(w * sizeof(unsigned int));
	Get_imgData(B, G, R, imgData, &fileHeader, &infoHeader, 2);

	//Show_res(B, "imgData_B.txt", infoHeader.biHeight + 4, infoHeader.biWidth + 4);
	//Show_res(G, "imgData_G.txt", infoHeader.biHeight + 4, infoHeader.biWidth + 4);
	//Show_res(R, "imgData_R.txt", infoHeader.biHeight + 4, infoHeader.biWidth + 4);

	/* 提前计算查表所需数据 */
	double littleGauss[6] = { 0.01441881 ,0.02808402,0.0350727,0.0547002,0.06831229,0.08531173 };
	double* table = (double*)malloc(6 * 256 * sizeof(double));	// 为table申请内存(6*256二维数组)
	if (!table) {
		printf("内存申请失败！\n");
		return 0;
	}
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 256; ++j) {
			table[i * 256 + j] = littleGauss[i] * j;
		}
	}

	/* 卷积 */
	unsigned char* convData = (unsigned char*)malloc((infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight * sizeof(char));
	if (!convData) {
		printf("内存申请失败！\n");
		return 0;
	}
	double startTime = omp_get_wtime();	// 开始计时
	printf("start: %f\n", startTime);

	/* 并行部分 */
	int i = convR, j = convR;
	int t1 = infoHeader.biHeight + convR, t2 = infoHeader.biWidth + convR;
#pragma omp parallel for num_threads(4) private(i,j)
	for (i = convR; i < t1; ++i)
		for (j = convR; j < t2; ++j) {
			register int i1 = i - 2, i2 = i - 1, i3 = i + 1, i4 = i + 2, j1 = j - 2, j2 = j - 1, j3 = j + 1, j4 = j + 2;
			register int cnt = ((i - convR) * infoHeader.biWidth + (j - convR)) * 3;
			convData[cnt] = (unsigned char)(
				table[B[i1][j1]] +
				table[256 + B[i1][j2]] +
				table[512 + B[i1][j]] +
				table[256 + B[i1][j3]] +
				table[B[i1][j4]] +
				table[256 + B[i2][j1]] +
				table[768 + B[i2][j2]] +
				table[1024 + B[i2][j]] +
				table[768 + B[i2][j3]] +
				table[256 + B[i2][j4]] +
				table[512 + B[i][j1]] +
				table[1024 + B[i][j2]] +
				table[1280 + B[i][j]] +
				table[1024 + B[i][j3]] +
				table[512 + B[i][j4]] +
				table[256 + B[i3][j1]] +
				table[768 + B[i3][j2]] +
				table[1024 + B[i3][j]] +
				table[768 + B[i3][j3]] +
				table[256 + B[i3][j4]] +
				table[B[i4][j1]] +
				table[256 + B[i4][j2]] +
				table[512 + B[i4][j]] +
				table[256 + B[i4][j3]] +
				table[B[i4][j4]]);
			convData[cnt + 1] = (unsigned char)(
				table[G[i1][j1]] +
				table[256 + G[i1][j2]] +
				table[512 + G[i1][j]] +
				table[256 + G[i1][j3]] +
				table[G[i1][j4]] +
				table[256 + G[i2][j1]] +
				table[768 + G[i2][j2]] +
				table[1024 + G[i2][j]] +
				table[768 + G[i2][j3]] +
				table[256 + G[i2][j4]] +
				table[512 + G[i][j1]] +
				table[1024 + G[i][j2]] +
				table[1280 + G[i][j]] +
				table[1024 + G[i][j3]] +
				table[512 + G[i][j4]] +
				table[256 + G[i3][j1]] +
				table[768 + G[i3][j2]] +
				table[1024 + G[i3][j]] +
				table[768 + G[i3][j3]] +
				table[256 + G[i3][j4]] +
				table[G[i4][j1]] +
				table[256 + G[i4][j2]] +
				table[512 + G[i4][j]] +
				table[256 + G[i4][j3]] +
				table[G[i4][j4]]);
			convData[cnt + 2] = (unsigned char)(
				table[R[i1][j1]] +
				table[256 + R[i1][j2]] +
				table[512 + R[i1][j]] +
				table[256 + R[i1][j3]] +
				table[R[i1][j4]] +
				table[256 + R[i2][j1]] +
				table[768 + R[i2][j2]] +
				table[1024 + R[i2][j]] +
				table[768 + R[i2][j3]] +
				table[256 + R[i2][j4]] +
				table[512 + R[i][j1]] +
				table[1024 + R[i][j2]] +
				table[1280 + R[i][j]] +
				table[1024 + R[i][j3]] +
				table[512 + R[i][j4]] +
				table[256 + R[i3][j1]] +
				table[768 + R[i3][j2]] +
				table[1024 + R[i3][j]] +
				table[768 + R[i3][j3]] +
				table[256 + R[i3][j4]] +
				table[R[i4][j1]] +
				table[256 + R[i4][j2]] +
				table[512 + R[i4][j]] +
				table[256 + R[i4][j3]] +
				table[R[i4][j4]]);
		}

	double endTime = omp_get_wtime();	// 停止计时
	printf("end: %f\n", endTime);
	printf("并行部分运行时间: %15.15f\n", endTime - startTime);

	/* 输出卷积后的图片 */
	Write_bmp("Open.bmp", convData, &fileHeader, &infoHeader);

	/* 输出卷积后的像素矩阵
	if (myid == 0) {
		FILE* fp;
		fp = fopen("result.txt", "wb");
		if (!fp) {
			printf("结果写入失败！\n");
			return 0;
		}

		int l = (infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight;
		for (int i = 0; i < l; ++i) {
			//printf("%d ", (int)gathered[i]);
			fprintf(fp, "%d ", (int)gathered[i]);
			if (i % 3 == 2)
				//printf("\n");
				fprintf(fp, "\n");
		}
		fclose(fp);
	} */

	/* 释放内存 */
	free(imgData);
	for (int i = 0; i < h; ++i)
		free(B[i]);
	free(B);
	for (int i = 0; i < h; ++i)
		free(G[i]);
	free(G);
	for (int i = 0; i < h; ++i)
		free(R[i]);
	free(R);

	return 0;
}
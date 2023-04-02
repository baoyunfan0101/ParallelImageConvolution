#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <malloc.h>
#include "omp.h"

/* �ļ���Ϣͷ�ṹ�� */
typedef struct tagBITMAPFILEHEADER
{
	unsigned short bfType;        // BM�ַ���, bmp��ʽ�ļ�:0x4d42(19778)
	unsigned int   bfSize;        // �ļ���С,��λ:�ֽ�(2-5�ֽ�)
	unsigned short bfReserved1;   // ����,0(6-7�ֽ�)
	unsigned short bfReserved2;   // ����,0(8-9�ֽ�)
	unsigned int   bfOffBits;     // ���ļ�ͷ���������ݵ�ƫ��(10-13�ֽ�)
} BITMAPFILEHEADER;

/* ͼ����Ϣͷ�ṹ�� */
typedef struct tagBITMAPINFOHEADER
{
	unsigned int    biSize;          // �˽ṹ��Ĵ�С(14-17�ֽ�)
	unsigned int    biWidth;         // ͼ��Ŀ�(18-21�ֽ�)
	unsigned int    biHeight;        // ͼ��ĸ�(22-25�ֽ�)
	unsigned short  biPlanes;        // bmpͼƬ��λ����,��ɫ�������,1(26-27�ֽ�)
	unsigned short  biBitCount;      // һ������ռ��λ��,һ��Ϊ24(28-29�ֽ�)
	unsigned int    biCompression;   // ͼ������ѹ��������,��ѹ��:0(30-33�ֽ�)
	unsigned int    biSizeImage;     // ����������ռ��С,bfSize-bfOffBits(34-37�ֽ�)
	unsigned int    biXPelsPerMeter; // ˮƽ�ֱ���,����/��,һ��Ϊ0(38-41�ֽ�)
	unsigned int    biYPelsPerMeter; // ��ֱ�ֱ���,����/��,һ��Ϊ0(42-45�ֽ�)
	unsigned int    biClrUsed;       // λͼʵ��ʹ�ò�ɫ���е���ɫ������,ʹ�����е�ɫ����:0(46-49�ֽ�)
	unsigned int    biClrImportant;  // ��ͼ����ʾ����ҪӰ�����ɫ��������Ŀ,����Ҫ:0(50-53�ֽ�)
} BITMAPINFOHEADER;

/* ��ɫ�� */
typedef struct RGBQuad
{
	unsigned char rgbBlue;		// ����ɫ����ɫ����,0-255
	unsigned char rgbGreen;		// ����ɫ����ɫ����,0-255
	unsigned char rgbRed;		// ����ɫ�ĺ�ɫ����,0-255
	unsigned char rgbReserved;	// ����,0
} RGBQuad;

/* չʾBITMAPFILEHEADER,������ */
void showBmpHead(BITMAPFILEHEADER bmf)
{
	printf("bfSize: %dkb\n", bmf.bfSize / 1024);
	printf("bfOffBits: %d\n", bmf.bfOffBits);
}

/* չʾBITMAPINFOHEADER,������ */
void showBmpInfoHead(BITMAPINFOHEADER bmi)
{
	printf("bfSize: %d\n", bmi.biSize);
	printf("biWidth: %d\n", bmi.biWidth);
	printf("biHeight: %d\n", bmi.biHeight);
	printf("biPlanes: %d\n", bmi.biPlanes);
	printf("biBitCount: %d\n", bmi.biBitCount);
}

/* ��ȡbmp�ļ�,������������ */
char* Read_bmp(char* filepath, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi)
{
	unsigned char* imgData;
	FILE* fp;

	fp = fopen(filepath, "rb");
	if (!fp) {
		printf("bmp�ļ���ʧ�ܣ�\n");
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
		printf("�ڴ�����ʧ�ܣ�\n");
		return NULL;
	}
	fseek(fp, bmf->bfOffBits, SEEK_SET);	// �ƶ��������ݿ�ʼλ��

	if (fread(imgData, (bitCount / (8 * sizeof(char))) * width * height * sizeof(char), 1, fp) != 1) {
		free(imgData);
		fclose(fp);
		printf("bmp�ļ��𻵣�\n");
		return NULL;
	}

	fclose(fp);
	return imgData;
}

/* д��bmp�ļ� */
void Write_bmp(char* filepath, unsigned char* imgData, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi)
{
	FILE* fp;
	long height = bmi->biHeight;
	unsigned int dwLineBytes = (bmi->biBitCount / (8 * sizeof(char))) * bmi->biWidth;
	fp = fopen(filepath, "wb");
	if (!fp) {
		printf("bmp�ļ�д��ʧ�ܣ�\n");
		return;
	}

	fwrite(bmf, sizeof(unsigned short), 1, fp);
	fwrite(&(bmf->bfSize), sizeof(BITMAPFILEHEADER) - 4, 1, fp);
	fwrite(bmi, sizeof(BITMAPINFOHEADER), 1, fp);

	/* 24λ���ɫͼ���޵�ɫ��
	RGBQuad* IpRGBQuad = (RGBQuad*)malloc(1 * sizeof(RGBQuad));
	IpRGBQuad[0].rgbRed = 0;
	IpRGBQuad[0].rgbGreen = 0;
	IpRGBQuad[0].rgbBlue = 0;
	IpRGBQuad[0].rgbReserved = 0;
	fwrite(IpRGBQuad, sizeof(RGBQuad), 1, fp); */

	fwrite(imgData, dwLineBytes * height, 1, fp);
}

/* ��ȡ��������(B,G,R�ֱ����) */
void Get_imgData(unsigned int** B, unsigned int** G, unsigned int** R,
	unsigned char* imgData, BITMAPFILEHEADER* bmf, BITMAPINFOHEADER* bmi,
	int convR)	// convRΪ����뾶
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

/* չʾ���ؾ���,��д���ļ� */
void Show_res(int** a, char* filepath, int h, int w)
{
	FILE* fp;
	fp = fopen(filepath, "wb");
	if (!fp) {
		printf("���д��ʧ�ܣ�\n");
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

/* ��˹�� */
double GaussCore[5][5] = {
	{0.01441881, 0.02808402, 0.0350727, 0.02808402, 0.01441881},
	{0.02808402, 0.0547002, 0.06831229, 0.0547002, 0.02808402},
	{0.0350727, 0.06831229, 0.08531173, 0.06831229, 0.0350727},
	{0.02808402, 0.0547002, 0.06831229, 0.0547002, 0.02808402},
	{0.01441881, 0.02808402, 0.0350727, 0.02808402, 0.01441881}
};

/* ������ */
int main(int argc, char** argv)
{
	/* ��ȡbmp�ļ� */
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;
	unsigned char* imgData = Read_bmp("data.bmp", &fileHeader, &infoHeader);
	if (!imgData)
		return 0;
	else if (infoHeader.biBitCount != 24) {
		printf("�ݲ�֧�ַ����ɫͼ��\n");
		return 0;
	}

	/* ��ȡ�������� */
	int convR = 2;	// ����˰뾶
	int h = infoHeader.biHeight + 2 * convR;
	int w = infoHeader.biWidth + 2 * convR;

	/* ��B,G,R��ȡΪ��ά���� */
	unsigned int** B = (unsigned int**)malloc(h * sizeof(unsigned int*));	// ΪB,G,R����ռ�(��ά����)
	unsigned int** G = (unsigned int**)malloc(h * sizeof(unsigned int*));
	unsigned int** R = (unsigned int**)malloc(h * sizeof(unsigned int*));
	if (!B || !G || !R) {
		printf("�ڴ�����ʧ�ܣ�\n");
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

	/* ��ǰ�������������� */
	double littleGauss[6] = { 0.01441881 ,0.02808402,0.0350727,0.0547002,0.06831229,0.08531173 };
	double* table = (double*)malloc(6 * 256 * sizeof(double));	// Ϊtable�����ڴ�(6*256��ά����)
	if (!table) {
		printf("�ڴ�����ʧ�ܣ�\n");
		return 0;
	}
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 256; ++j) {
			table[i * 256 + j] = littleGauss[i] * j;
		}
	}

	/* ��� */
	unsigned char* convData = (unsigned char*)malloc((infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight * sizeof(char));
	if (!convData) {
		printf("�ڴ�����ʧ�ܣ�\n");
		return 0;
	}
	double startTime = omp_get_wtime();	// ��ʼ��ʱ
	printf("start: %f\n", startTime);

	/* ���в��� */
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

	double endTime = omp_get_wtime();	// ֹͣ��ʱ
	printf("end: %f\n", endTime);
	printf("���в�������ʱ��: %15.15f\n", endTime - startTime);

	/* ���������ͼƬ */
	Write_bmp("Open.bmp", convData, &fileHeader, &infoHeader);

	/* ������������ؾ���
	if (myid == 0) {
		FILE* fp;
		fp = fopen("result.txt", "wb");
		if (!fp) {
			printf("���д��ʧ�ܣ�\n");
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

	/* �ͷ��ڴ� */
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
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <malloc.h>
#include "mpi.h"

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

	/* ���в��� */
	int myid, numprocs, nameelen;	// ��ǰ���̱��,������,���������Ƴ���
	char processer_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processer_name, &nameelen);
	double timestart = MPI_Wtime();	// ��ʱ����ʼ

	/* ���㵱ǰ���̿�ʼ�ͽ�������λ��(instead of�ַ�) */
	int start = (infoHeader.biHeight / numprocs) * myid + convR;
	int end;
	if (myid == numprocs - 1)
		end = infoHeader.biHeight + convR;
	else
		end = (infoHeader.biHeight / numprocs) * (myid + 1) + convR;
	int th = end - start;	// ����(��ʱ����)
	int tw = infoHeader.biWidth + convR;	// ��������λ��(��ʱ����)

	/* ��� */
	unsigned char* convData = (unsigned char*)malloc((infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight * sizeof(char));
	if (!convData) {
		printf("�ڴ�����ʧ�ܣ�\n");
		return 0;
	}
	int cnt = 0;
	for (int i = start; i < end; ++i)
		for (int j = convR; j < tw; ++j) {
			register int i1 = i - 2, i2 = i - 1, i3 = i + 1, i4 = i + 2, j1 = j - 2, j2 = j - 1, j3 = j + 1, j4 = j + 2;
			convData[cnt++] =
				GaussCore[0][0] * B[i1][j1] +
				GaussCore[0][1] * B[i1][j2] +
				GaussCore[0][2] * B[i1][j] +
				GaussCore[0][3] * B[i1][j3] +
				GaussCore[0][4] * B[i1][j4] +
				GaussCore[1][0] * B[i2][j1] +
				GaussCore[1][1] * B[i2][j2] +
				GaussCore[1][2] * B[i2][j] +
				GaussCore[1][3] * B[i2][j3] +
				GaussCore[1][4] * B[i2][j4] +
				GaussCore[2][0] * B[i][j1] +
				GaussCore[2][1] * B[i][j2] +
				GaussCore[2][2] * B[i][j] +
				GaussCore[2][3] * B[i][j3] +
				GaussCore[2][4] * B[i][j4] +
				GaussCore[3][0] * B[i3][j1] +
				GaussCore[3][1] * B[i3][j2] +
				GaussCore[3][2] * B[i3][j] +
				GaussCore[3][3] * B[i3][j3] +
				GaussCore[3][4] * B[i3][j4] +
				GaussCore[4][0] * B[i4][j1] +
				GaussCore[4][1] * B[i4][j2] +
				GaussCore[4][2] * B[i4][j] +
				GaussCore[4][3] * B[i4][j3] +
				GaussCore[4][4] * B[i4][j4];
			convData[cnt++] =
				GaussCore[0][0] * G[i1][j1] +
				GaussCore[0][1] * G[i1][j2] +
				GaussCore[0][2] * G[i1][j] +
				GaussCore[0][3] * G[i1][j3] +
				GaussCore[0][4] * G[i1][j4] +
				GaussCore[1][0] * G[i2][j1] +
				GaussCore[1][1] * G[i2][j2] +
				GaussCore[1][2] * G[i2][j] +
				GaussCore[1][3] * G[i2][j3] +
				GaussCore[1][4] * G[i2][j4] +
				GaussCore[2][0] * G[i][j1] +
				GaussCore[2][1] * G[i][j2] +
				GaussCore[2][2] * G[i][j] +
				GaussCore[2][3] * G[i][j3] +
				GaussCore[2][4] * G[i][j4] +
				GaussCore[3][0] * G[i3][j1] +
				GaussCore[3][1] * G[i3][j2] +
				GaussCore[3][2] * G[i3][j] +
				GaussCore[3][3] * G[i3][j3] +
				GaussCore[3][4] * G[i3][j4] +
				GaussCore[4][0] * G[i4][j1] +
				GaussCore[4][1] * G[i4][j2] +
				GaussCore[4][2] * G[i4][j] +
				GaussCore[4][3] * G[i4][j3] +
				GaussCore[4][4] * G[i4][j4];
			convData[cnt++] =
				GaussCore[0][0] * R[i1][j1] +
				GaussCore[0][1] * R[i1][j2] +
				GaussCore[0][2] * R[i1][j] +
				GaussCore[0][3] * R[i1][j3] +
				GaussCore[0][4] * R[i1][j4] +
				GaussCore[1][0] * R[i2][j1] +
				GaussCore[1][1] * R[i2][j2] +
				GaussCore[1][2] * R[i2][j] +
				GaussCore[1][3] * R[i2][j3] +
				GaussCore[1][4] * R[i2][j4] +
				GaussCore[2][0] * R[i][j1] +
				GaussCore[2][1] * R[i][j2] +
				GaussCore[2][2] * R[i][j] +
				GaussCore[2][3] * R[i][j3] +
				GaussCore[2][4] * R[i][j4] +
				GaussCore[3][0] * R[i3][j1] +
				GaussCore[3][1] * R[i3][j2] +
				GaussCore[3][2] * R[i3][j] +
				GaussCore[3][3] * R[i3][j3] +
				GaussCore[3][4] * R[i3][j4] +
				GaussCore[4][0] * R[i4][j1] +
				GaussCore[4][1] * R[i4][j2] +
				GaussCore[4][2] * R[i4][j] +
				GaussCore[4][3] * R[i4][j3] +
				GaussCore[4][4] * R[i4][j4];
		}

	/* �ۼ� */
	int datatype_size;
	MPI_Type_size(MPI_INT, &datatype_size);
	int* length = (int*)malloc(datatype_size * numprocs);	// ÿ���������ݳ���
	int* displs = (int*)malloc(datatype_size * numprocs);	// ÿ������������ʼλ��
	// �ۼ�����
	MPI_Type_size(MPI_CHAR, &datatype_size);
	unsigned char* gathered = (unsigned char*)malloc((infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight * datatype_size);
	if (!length || !displs || !gathered) {
		printf("�ڴ�����ʧ�ܣ�\n");
		return 0;
	}

	if (myid == 0) {	// ������
		displs[0] = 0;
		for (int myid = 0; myid < numprocs; ++myid) {	// ����ÿ�����̵����ݳ���,������ʼλ��
			int start = (infoHeader.biHeight / numprocs) * myid + convR;
			int end;
			if (myid == numprocs - 1)
				end = infoHeader.biHeight + convR;
			else
				end = (infoHeader.biHeight / numprocs) * (myid + 1) + convR - 1;
			int th = end - start;

			length[myid] = (infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight;
			if (myid > 0)
				displs[myid] = length[myid - 1] + displs[myid - 1];
		}

		MPI_Type_size(MPI_CHAR, &datatype_size);
		gathered = (unsigned char*)malloc((infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight * datatype_size);
		if (!gathered) {
			printf("�ڴ�����ʧ�ܣ�\n");
			return 0;
		}
	}

	printf("%d#���̴�������: %d\n", myid, (infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * th);
	MPI_Gatherv(convData, (infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * th, MPI_CHAR, gathered, length, displs, MPI_CHAR, 0, MPI_COMM_WORLD);

	/* ���㲢�в�������ʱ�� */
	double timeend = MPI_Wtime();
	printf("%d#���̲��в�������ʱ��: %lfs\n", myid, timeend - timestart);

	/* �����ӽ���������ڴ� */
	if (myid != 0)
		free(convData);
	MPI_Finalize();

	/* ���������ͼƬ */
	Write_bmp("MPI.bmp", gathered, &fileHeader, &infoHeader);

	/* ������������ؾ��� */
	if (myid == 0) {
		FILE* fp;
		fp = fopen("result.txt", "wb");
		if (!fp) {
			printf("���д��ʧ�ܣ�\n");
			return 0;
		}

		int tmp = 3 * 4096;
		int l = (infoHeader.biBitCount / (8 * sizeof(char))) * infoHeader.biWidth * infoHeader.biHeight;
		for (int i = 0; i < l; ++i) {
			fprintf(fp, "%d ", (int)gathered[i]);
			if (i % tmp == tmp - 1)
				fprintf(fp, "\n");
		}
		fclose(fp);
	}

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
#include <stdio.h>
#include <cuda.h>
#include "kernel.cu"
#include <time.h>

#define m(y,x) mapa[(y * cols) + x]

/*Definición de constantes*/
#define currentGPU 0		//El número más alto suele indicar la salida de vídeo
#define MAX 50

/*Declaración de funciones  CPU*/

int main()
{
	int rows, cols, i;
	rows = 5;
	cols = 5;

	/*Declaración e inicialización de variables CPU (HOST)*/
	int *mapa;

	/*Indicamos la GPU (DEVICE) que vamos a utilizar*/
	int *mapad;

	/*Reserva de memoria para de variables CPU*/
	if ((mapa=(int *) calloc(rows * cols, sizeof(int)))==NULL){		//mapa
		printf("error\n");
		exit (-1);
	}
	


	/*Reserva de memoria para variables del DEVICE (en memoria GPU)*/
	
	cudaMalloc( (void**) &mapad, sizeof(int) * (int) rows * cols);
	

	/*Inicialización del mapa*/
	int size = rows * cols;
	int tam = (int) ceil( ((float)(rows * cols))/size);
	dim3 bloqdiminit(128,1);
	dim3 griddiminit(tam,1);

	gpu_init<<<griddiminit,bloqdiminit>>>(mapad,MAX,size);

	/*Copia de datos del HOST al DEVICE*/
	cudaMemcpy(mapa,mapad,sizeof(int) * rows*cols,cudaMemcpyDeviceToHost);

	/*Lanzamos la función del DEVICE*/
	cudaDeviceSynchronize();
	for (i = 0; i<5*5;i++){
		printf(" %d \n",mapa[i]);
	}

	/*Liberamos memoria del DEVICE*/
	cudaFree(mapad);

	/*Liberamos memoria del HOST*/
	free(mapa);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();
} //main





#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define m(y,x) mapa[(y * cols) + x]

/*Definición de constantes*/
#define currentGPU 0		//El número más alto suele indicar la salida de vídeo
#define MAX 50

typedef struct {
	int y;
	int x;
} Antena;

__global__ void gpu_init(int *mapad, int max, int size)
{
	/*Identificaciones necesarios*/
	int IDX_Thread = threadIdx.x; 
	int IDY_Thread = threadIdx.y;
	int IDX_block =	blockIdx.x;
	int IDY_block =	blockIdx.y;
	int shapeGrid_X = gridDim.x;
	int threads_per_block =	blockDim.x * blockDim.y;
	int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);

	if (position<size) mapad[position] = max;
}

__global__ void gpu_actualizar(int *mapad, int rows, int cols, Antena antena, int size)
{
	int IDX_Thread = threadIdx.x; 
	int IDY_Thread = threadIdx.y;
	int IDX_block =	blockIdx.x;
	int IDY_block =	blockIdx.y;
	int shapeGrid_X = gridDim.x;
	int threads_per_block =	blockDim.x * blockDim.y;
	int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);

	if(position<size)
	{
		int x,y;
		y=(int)position/cols;
		x=position-y*rows;
		int dist = abs(antena.x -x) + abs(antena.y - y);
		int nuevadist = dist*dist;
		if(nuevadist<mapad[position])
		{
			mapad[position] = nuevadist;
		}
	}

}
int manhattan(Antena a, int y, int x){

	int dist = abs(a.x -x) + abs(a.y - y);
	return dist * dist;
}


int main()
{
	int rows, cols,nAntenas, i,j;
	rows = 5;
	cols = 5;
	nAntenas = 1;
	Antena *antenas;

	 if ((antenas = (Antena *) calloc(rows * cols, sizeof(Antena)))==NULL){		//mapa
		printf("error\n");
		exit (-1);
	}

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
	printf("matriz:\n");
	for (i = 0; i<5;i++){
		for (j=0;j<5;j++){
			printf(" %d ",mapa[j*5+i]);
		}
		printf("\n");

	}
	printf("fin de la matriz\n");

	Antena ant = {2,2};

	gpu_actualizar<<<griddiminit,bloqdiminit>>>(mapad, rows,  cols,  ant,  size);
	
	cudaMemcpy(mapa,mapad,sizeof(int) * rows*cols,cudaMemcpyDeviceToHost);

	/*Lanzamos la función del DEVICE*/
	cudaDeviceSynchronize();
	printf("matriz2:\n");
	for (i = 0; i<5;i++){
		for (j=0;j<5;j++){
			printf(" %d ",mapa[j*5+i]);
		}
		printf("\n");

	}
	printf("fin de la matriz2\n");

	/*Liberamos memoria del DEVICE*/
	cudaFree(mapad);

	/*Liberamos memoria del HOST*/
	free(mapa);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();
} //main





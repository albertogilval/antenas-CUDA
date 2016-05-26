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

int calcular_max(int * mapa, int rows, int cols){

	int max = 0;

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			if(m(i,j)>max){
				max = m(i,j);			
			}

		} // j
	} // i

	return max;
}

Antena nueva_antena(int * mapa, int rows, int cols, int min){

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			if(m(i,j)==min){

				Antena antena = {i,j};
				return antena;
			}

		} // j
	} // i
}

int main(int nargs, char ** vargs)
{

	//
	// 1. LEER DATOS DE ENTRADA
	//

	// Comprobar número de argumentos
	if(nargs < 7){
		fprintf(stderr,"Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",vargs[0]);
		return -1;
	}

	// Leer los argumentos de entrada
	int rows = atoi(vargs[1]);
	int cols = atoi(vargs[2]);
	int distMax = atoi(vargs[3]);
	int nAntenas = atoi(vargs[4]);	

	int i,j;

	Antena *antenas;

	 if ((antenas = (Antena *) calloc(rows * cols, sizeof(Antena)))==NULL){		//mapa
		printf("error\n");
		exit (-1);
	}

	// Leer antenas
	for(int i=0; i<nAntenas; i++){
		antenas[i].x = atoi(vargs[5+i*2]);
		antenas[i].y = atoi(vargs[6+i*2]);

		if(antenas[i].y<0 || antenas[i].y>=rows || antenas[i].x<0 || antenas[i].x>=cols ){
			fprintf(stderr,"Antena #%d está fuera del mapa\n",i);
			return -1;
		}
	}	

	//
	// 2. INICIACIÓN
	//

	// Medir el tiempo
	

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
	cudaDeviceSynchronize();

	/*Copia de datos del HOST al DEVICE*/
	cudaMemcpy(mapa,mapad,sizeof(int) * rows*cols,cudaMemcpyDeviceToHost);

	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		gpu_actualizar<<<griddiminit,bloqdiminit>>>(mapad, rows,  cols,  antenas[i],  size);
		cudaDeviceSynchronize();
	}
	
	/*Copia de datos del DEVICE al HOST*/
	cudaMemcpy(mapa,mapad,sizeof(int) * rows*cols,cudaMemcpyDeviceToHost);

	/*Lanzamos la función del DEVICE*/
	cudaDeviceSynchronize();
	printf("matriz:\n");
	for (i = 0; i<rows;i++){
		for (j=0;j<cols;j++){
			printf(" %d ",mapa[j*5+i]);
		}
		printf("\n");

	}
	printf("fin de la matriz\n----\n");

	// Contador de antenas
	int nuevas = 0;

	while(1){

		//calcular el maximo
		int max = calcular_max(mapa, rows, cols);
		printf("max: %d\n",max);

		// Salimos si ya hemos cumplido el maximo
		if (max <= distMax) break;
		
		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualización del mapa
		Antena antena = nueva_antena(mapa, rows, cols, max);
		gpu_actualizar<<<griddiminit,bloqdiminit>>>(mapad, rows,  cols,  antena,  size);
		cudaDeviceSynchronize();
		cudaMemcpy(mapa,mapad,sizeof(int) * rows*cols,cudaMemcpyDeviceToHost);

		printf("\n");
		for (i = 0; i<rows;i++){
			for (j=0;j<cols;j++){
				printf(" %d ",mapa[j*5+i]);
			}
		printf("\n");

	}
	printf("\n");

	}
	
	cudaDeviceSynchronize();
	printf("-----\nmatriz final:\n");
	for (i = 0; i<rows;i++){
		for (j=0;j<cols;j++){
			printf(" %d ",mapa[j*5+i]);
		}
		printf("\n");

	}
	printf("fin de la matriz final\n");
	

	/*Liberamos memoria del DEVICE*/
	cudaFree(mapad);

	/*Liberamos memoria del HOST*/
	free(mapa);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();
} //main





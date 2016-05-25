#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <limits.h>

// Include para las utilidades de computación paralela
#include "cputils.h"

/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
} Antena;
/**
 * Macro para acceder a las posiciones del mapa
 */
#define m(y,x) mapa[ (y * cols) + x ]

__global__ void gpu_init(int *mapad, int INT_MAX)
{
	/*Identificaciones necesarios*/
	int IDX_Thread = threadIdx.x; //Identificacion del hilo en la dimension
	int IDY_Thread = threadIdx.y; //Identificacion del hilo en la dimension y
	int IDX_block =	blockIdx.x; //Identificacion del bloque en la dimension x
	int IDY_block = blockIdx.y; //Identificacion del bloque en la dimension y
	int shapeGrid_X = gridDim.x; //Numeros del bloques en la dimension x
	int threads_per_block =	blockDim.x * blockDim.y; //Numero de hilos por bloque (1 dimension)

	/*Formula para calcular la posicion*/	//Posicion del vector dependiendo del hilo y del bloque 
	int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);

	//inicializamos
	mapad[position] = INT_MAX;
}

/**
 * Función principal
 */
int main(int nargs, char ** vargs){


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

	if(nAntenas<1 || nargs != (nAntenas*2+5)){
		fprintf(stderr,"Error en la lista de antenas\n");
		return -1;
	}


	// Mensaje
	printf("Calculando el número de antenas necesarias para cubrir un mapa de"
		   " (%d x %d)\ncon una distancia máxima no superior a %d "
		   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);

	// Reservar memoria para las antenas
	
	// Leer antenas
	


	//
	// 2. INICIACIÓN
	//

	// Medir el tiempo
	double tiempo = cp_Wtime();

	// Crear el mapa
	int * mapa = malloc((size_t) (rows*cols) * sizeof(int) );
	//Crear y reservar la memoria DEVICE
	int *mapad;
	cudaMalloc( (void**) &mapad, sizeof(int) * (int) (rows*cols));

	// Iniciar el mapa con el valor MAX INT
//paralelizar
	int tam = (((rows*cols)+512-1)/512);
	dim3 bloqdim(512,1);
	dim3 griddim(tam,1);

	gpu_init<<<griddim, bloqdim>>>(mapad,INT_MAX);
	cudaDeviceSynchronize();//no se si es necesario (creo que no)
	cudaMemcpy(mapad, mapa, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);

//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;	

	// Salida
	printf("Time: %f\n",tiempo);

	cudaFree(mapad);
	return 0;
}



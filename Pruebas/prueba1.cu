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

#define currentGPU 0

__global__ void gpu_init(int *mapad, int max, int size){
	/*Identificaciones necesarios*/
	int IDX_Thread = threadIdx.x;	/*Identificacion del hilo en la dimension*/
	int IDY_Thread = threadIdx.y;	/*Identificacion del hilo en la dimension y*/
	int IDX_block =	blockIdx.x;	/*Identificacion del bloque en la dimension x*/
	int IDY_block = blockIdx.y;	/*Identificacion del bloque en la dimension y */
	int shapeGrid_X = gridDim.x; 	/*Numeros del bloques en la dimension */ 

	int threads_per_block =	blockDim.x * blockDim.y; /* Numero de hilos por bloque (1 dimension) */

	/*Formula para calcular la posicion*/	//Posicion del vector dependiendo del hilo y del bloque 
	int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);

	//inicializamos
	if(position<size) mapad[position] = max;
}

void print_mapa(int * mapa, int rows, int cols, Antena * a){


	if(rows > 50 || cols > 30){
		printf("Mapa muy grande para imprimir\n");
		return;
	};


	printf("Mapa [%d,%d]\n",rows,cols);
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			int val = m(i,j);
			printf(" %6d ",val);

		}
		printf("\n");
	}
	printf("\n");
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
	int * mapa = (int *) malloc((size_t) (rows*cols) * sizeof(int) );
	
	//Crear y reservar la memoria DEVICE
	int *mapad;
	cudaMalloc( (void**) &mapad, sizeof(int) * (int) (rows*cols));

	// Iniciar el mapa con el valor MAX INT
	
	int size = rows*cols;	
	int tam = (int) ceil( ((float)(rows * cols)) /size);
	dim3 bloqdimfunc1(128,1);
	dim3 griddimfunc1(tam,1);
	/* Enviamos la matriz al dispositivo */
	cudaMemcpy(mapad, mapa, sizeof(int) * (rows*cols),cudaMemcpyHostToDevice);
	
	/* Llamamos a la funcion gpu_init */
	gpu_init<<<griddimfunc1, bloqdimfunc1>>>(mapad,INT_MAX,size);
	
	/* Sincronizamos para estabilizar los datos */
	cudaDeviceSynchronize();
	
	/* Recibimos la matriz de Device */
	cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);
	print_mapa(mapa,rows,cols,NULL);
	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;	

	// Salida
	printf("Time: %f\n",tiempo);
	
	/* Comprobamos si se ha realizado bien la funcion */
	
	int error=0,z;
	for(z=0;z<rows*cols;z++){
		if(mapa[z]!=INT_MAX) error=1;
	}
	if(error==1) printf("Algo salio mal\n");
	else printf ("Todo correcto\n");
	
	
	/* Liberamos memoria */
	cudaFree(mapad);
	
	/* Liberamos el dispositivo */
	cudaDeviceReset();
	return 0;
}


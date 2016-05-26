/**
 * Computación Paralela (curso 1516)
 *
 * Alberto Gil
 * Guillermo Cebrian
 */


// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <limits.h>


// Include para las utilidades de computación paralela
#include "cputils.h"

/**
 * Macro para acceder a las posiciones del mapa
 */

#define m(y,x) mapa[ (y * cols) + x ]

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

/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
} Antena;


/**
 * Función de ayuda para imprimir el mapa
 */
/*
void print_mapa(int * mapa, int rows, int cols, Antena * a){


	if(rows > 50 || cols > 30){
		printf("Mapa muy grande para imprimir\n");
		return;
	};

	#define ANSI_COLOR_RED     "\x1b[31m"
	#define ANSI_COLOR_GREEN   "\x1b[32m"
	#define ANSI_COLOR_RESET   "\x1b[0m"

	printf("Mapa [%d,%d]\n",rows,cols);
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			int val = m(i,j);

			if(val == 0){
				if(a != NULL && a->x == j && a->y == i){
					printf( ANSI_COLOR_RED "   A"  ANSI_COLOR_RESET);
				} else { 
					printf("A");
				}
			} else {
				printf("%4d",val);
			}
		}
		printf("\n");
	}
	printf("\n");
}
*/
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
 * Distancia de una antena a un punto (y,x)
 * @note Es el cuadrado de la distancia para tener más carga
 */
int manhattan(Antena a, int y, int x){

	int dist = abs(a.x -x) + abs(a.y - y);
	return dist * dist;
}



/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar(int * mapa, int rows, int cols, Antena antena){

	m(antena.y,antena.x) = 0;

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			int nuevadist = manhattan(antena,i,j);

			if(nuevadist < m(i,j)){
				m(i,j) = nuevadist;
			}

		} // j
	} // i
}



/**
 * Calcular la distancia máxima en el mapa
 */
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


/**
 * Calcular la posición de la nueva antena
 */

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
	Antena * antenas = (Antena *) malloc(sizeof(Antena) * (size_t) nAntenas);
	if(!antenas){
		fprintf(stderr,"Error al reservar memoria para las antenas inicales\n");
		return -1;
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
	double tiempo = cp_Wtime();

	// Crear el mapa
	int * mapa =  (int *) malloc((size_t) (rows*cols) * sizeof(int) );
	
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
	
	printf("%d INTMAX\n",INT_MAX);
	/* Llamamos a la funcion gpu_init */
	gpu_init<<<griddimfunc1, bloqdimfunc1>>>(mapad,INT_MAX,size);
	
	/* Sincronizamos para estabilizar los datos */
	cudaDeviceSynchronize();
	
	/* Recibimos la matriz de Device */
	cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);

	print_mapa(mapa,rows,cols,NULL);

	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		actualizar(mapa,rows,cols,antenas[i]);
	}
	
	print_mapa(mapa,rows,cols,NULL);

	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	// Contador de antenas
	int nuevas = 0;
	
	while(1){

		// Calcular el máximo
		int max = calcular_max(mapa, rows, cols);

		// Salimos si ya hemos cumplido el maximo
		if (max <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualización del mapa
		Antena antena = nueva_antena(mapa, rows, cols, max);
		actualizar(mapa,rows,cols,antena);

	}

	print_mapa(mapa,rows,cols,NULL);

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;
	
	/* Liberamos memoria */
	cudaFree(mapad);
	
	/* Liberamos el dispositivo */
	cudaDeviceReset();

	// Salida
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);

	return 0;
}




/**
 * Computación Paralela (curso 1516)
 *
 * Colocación de antenas
 * Versión secuencial
 *
 * @author Javier
 */


// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "iniciar.cu"
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
__global__ void gpuFunc_vecAdd(int *mapad, int INT_MAX)
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
 * Función de ayuda para imprimir el mapa
 */
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
					printf( ANSI_COLOR_GREEN "   A"  ANSI_COLOR_RESET);
				}
			} else {
				printf("%4d",val);
			}
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
	Antena * antenas = malloc(sizeof(Antena) * (size_t) nAntenas);
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
 
/*	for(int i=0; i<(rows*cols); i++){
		mapa[i] = INT_MAX;
	}*/

	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		actualizar(mapa,rows,cols,antenas[i]);
	}

	// Transferencia a memoria Device
	cudaMemcpy(mapad,mapa,sizeof(int) * (rows*cols),cudaMemcpyHostToDevice);

	// Debug
#ifdef DEBUG
	print_mapa(mapa,rows,cols,NULL);
#endif


	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	// Contador de antenas
	int nuevas = 0;
	
	while(1){

		// Calcular el máximo
//paralelizar
		int max = calcular_max(mapa, rows, cols);

		// Salimos si ya hemos cumplido el maximo
		if (max <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualización del mapa
		Antena antena = nueva_antena(mapa, rows, cols, max);
		actualizar(mapa,rows,cols,antena);

	}

	// Debug
#ifdef DEBUG
	print_mapa(mapa,rows,cols,NULL);
#endif

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;	

	// Salida
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);

	return 0;
}




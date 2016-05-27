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

/**
 * Definimos el tamaño de bloque
 */
#define TAMBLOCK 128

/* Funcion que inicializa la matriz al valor maximo */

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

/* Actualiza el mapa despues de colocar una antena*/

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
		x=position-y*cols;
		int dist = abs(antena.x -x) + abs(antena.y - y);
		int nuevadist = dist*dist;
		if(nuevadist<mapad[position])
		{
			mapad[position] = nuevadist;
		}
	}
}


__global__ void gpu_nueva_antena(int *mapad, int rows, int cols, int max, Antena antenaD)
{
	__shared__ bool find = 0;
	int IDX_Thread = threadIdx.x; 
	int IDY_Thread = threadIdx.y;
	int IDX_block =	blockIdx.x;
	int IDY_block =	blockIdx.y;
	int shapeGrid_X = gridDim.x;
	int threads_per_block =	blockDim.x * blockDim.y;

	int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);
	int idglobal = ((gridDim.x*gridDim.y)*(IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);
	int size = rows*cols;	
	
	if(position<size && find==0)
	{
		//printf("position: %d\t idglobal: %d\n",position,idglobal);
		int x,y;
		y=(int)idglobal/cols;
		x=idglobal-y*cols;
		if(mapad[position]==max) {
			antenaD = (Antena){x,y};
			find=1;
			}
		printf("Antena [%d,%d]\n",antenaD.x,antenaD.y);
	}
	/*
	__shared__ int pos = size;
	int x,y;
	y=(int)idglobal/cols;
	x=idglobal-y*cols;
	if(mapad[position]==max && position<pos){
		pos=position;
		antenaD = (Antena){x,y};
	}
	*/
}

/**
 * Función de ayuda para imprimir el mapa
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
	
	Antena antenaD;
	cudaMalloc( (void**) &antenaD, sizeof(Antena));
	// Iniciar el mapa con el valor MAX INT

	int size = rows*cols;	
	int tam = (int) ceil((float)(size /TAMBLOCK))+1;
	printf("Tam: %d\n",tam);
	dim3 bloqdimmatriz(TAMBLOCK,1);
	dim3 griddimmatriz(tam,1);

	/* Enviamos la matriz al dispositivo */
	cudaMemcpy(mapad, mapa, sizeof(int) * (rows*cols),cudaMemcpyHostToDevice);
	
	/* Llamamos a la funcion gpu_init */
	gpu_init<<<griddimmatriz, bloqdimmatriz>>>(mapad,INT_MAX,size);
	
	/* Sincronizamos para estabilizar los datos */
	cudaDeviceSynchronize();

// Debug
	/* Recibimos la matriz de Device para imprimirla */
	cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);
	printf("Matriz inicializada\n");
	print_mapa(mapa,rows,cols,NULL);
// Debug fin

	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		gpu_actualizar<<<griddimmatriz,bloqdimmatriz>>>(mapad, rows,  cols,  antenas[i], size);
	}

	/* Recibimos la matriz de Device */
	cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);
	printf("Mapa con antenas iniciales\n");

// Debug ini

	print_mapa(mapa,rows,cols,NULL);

// Debug fin

	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	// Contador de antenas
	int nuevas = 0;
	
	while(1){

		// Calcular el máximo
		int max = calcular_max(mapa, rows, cols);
		printf("Max: %d distMax: %d\n",max,distMax);
		// Salimos si ya hemos cumplido el maximo
		if (max <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
print_mapa(mapa,rows,cols,NULL);
		// Calculo de la nueva antena y actualización del mapa
		// Creamos antena host	
		Antena antena;
		//Llamamos a la funcion mandando la antenaD
		gpu_nueva_antena<<<griddimmatriz,bloqdimmatriz>>>(mapad,rows,cols,max,antena);
		// Recibimos antenaD y la guardamos en antena
		cudaDeviceSynchronize();
		//printf("Antena : [%d,%d]\n",antena.y,antena.x)

		// Enviamos la matriz al device para que se actualice
		//cudaMemcpy(mapad, mapa, sizeof(int) * (rows*cols),cudaMemcpyHostToDevice);
		// Llamamos a la funcion
		gpu_actualizar<<<griddimmatriz,bloqdimmatriz>>>(mapad, rows,  cols,  antena, size);
		// De momento recibimos la matriz por que hay que calcular el maximo
		cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);
print_mapa(mapa,rows,cols,NULL);

	}


	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;
	printf("Mapa final\n");
	print_mapa(mapa,rows,cols,NULL);
	/* Liberamos memoria */
	cudaFree(mapad);
	
	/* Liberamos el dispositivo */
	cudaDeviceReset();

	// Salida
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);

	return 0;
}




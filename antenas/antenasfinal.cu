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

__global__ void gpu_reduce(int *c, int size)
{
	/*Identificaciones necesarios*/
	int IDX_Thread = threadIdx.x; 
	int IDY_Thread = threadIdx.y;
	int IDX_block =	blockIdx.x;
	int IDY_block =	blockIdx.y;
	int shapeGrid_X = gridDim.x;
	int threads_per_block =	blockDim.x * blockDim.y;
	int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);


	if(position<size){
		if(size%2 != 0)
		{
			if(c[position]<c[size-1])
			{
				c[position]=c[size-1];
			}
		}else{

			if(c[position]<c[position+size/2])
			{
				c[position]=c[position+size/2];
			}
		}
	}
}

int reduce(int *maximo,int *c, int *v, int size,dim3 bd,dim3 gd)
{
	int t=size;
	
		while(t!=1){
		
			gpu_reduce<<<gd,bd>>>(c,t);
			cudaDeviceSynchronize();
			if(t%2==0){
				t=t/2;
			}else{
				t -= 1;
			}
		}

		cudaMemcpy(maximo,c,sizeof(int) * 1,cudaMemcpyDeviceToHost);
	return maximo[0];
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

	/*Declaración e inicialización de variables CPU (HOST)*/
	int *mapa, *maximo;

	/*Indicamos la GPU (DEVICE) que vamos a utilizar*/
	int *mapad;

	/*Reserva de memoria para de variables CPU*/
	if ((mapa=(int *) calloc(rows * cols, sizeof(int)))==NULL){		//mapa
		printf("error\n");
		exit (-1);
	}
	
	if ((maximo=(int *) calloc(1, sizeof(int)))==NULL){
		printf("error\n");
		exit (-1);
	}

	/*Reserva de memoria para variables del DEVICE (en memoria GPU)*/
	
	cudaMalloc( (void**) &mapad, sizeof(int) * (int) rows * cols);
	
	/* Creamos las dimensiones de grid y bloques */

	int size = rows*cols;	
	int tam = (int) ceil((float)(size /TAMBLOCK))+1;
	dim3 bloqdimmatriz(TAMBLOCK,1);
	dim3 griddimmatriz(tam,1);

	/* Enviamos la matriz al dispositivo */
	cudaMemcpy(mapad, mapa, sizeof(int) * (rows*cols),cudaMemcpyHostToDevice);
	
	/* Llamamos a la funcion gpu_init */
	gpu_init<<<griddimmatriz, bloqdimmatriz>>>(mapad,INT_MAX,size);
	
	/* Sincronizamos para estabilizar los datos */
	cudaDeviceSynchronize();

/*Debug
	cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);
	printf("Matriz inicializada\n");
	print_mapa(mapa,rows,cols,NULL);
Debug fin*/

	// Colocar las antenas iniciales el mapa ya esta en device
	for(int i=0; i<nAntenas; i++){
		gpu_actualizar<<<griddimmatriz,bloqdimmatriz>>>(mapad, rows,  cols,  antenas[i], size);
	}
	cudaDeviceSynchronize();
	
	
/* Debug ini
	printf("Mapa con antenas iniciales\n");
	print_mapa(mapa,rows,cols,NULL);

Debug fin */

	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	// Contador de antenas
	int nuevas = 0;
	
	/* Recibimos la matriz de Device */
	cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);
	
	while(1){

		// Calcular el máximo
		int max = reduce(maximo,mapad,mapa, rows * cols,bloqdimmatriz,griddimmatriz);
		
		// Salimos si ya hemos cumplido el maximo
		if (max <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Traemos la matriz a device para calcular la siguiente antena
		cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);
		// Calculo de la nueva antena y actualización del mapa
		// Creamos la siguiente antena donde este el primer maximo
		Antena antena = nueva_antena(mapa,rows,cols,max);

		// Enviamos la matriz al device para que se actualice
		cudaMemcpy(mapad, mapa, sizeof(int) * (rows*cols),cudaMemcpyHostToDevice);

		// Llamamos a la funcion
		gpu_actualizar<<<griddimmatriz,bloqdimmatriz>>>(mapad, rows,  cols,  antena, size);
		//Sincronizamos para estar seguros de que se actualiza todo el mapa
		cudaDeviceSynchronize();
		
		// Recibimos la matriz para calcular el maximo
		cudaMemcpy(mapa, mapad, sizeof(int) * (rows*cols),cudaMemcpyDeviceToHost);

	}


	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;

	// Salida
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);

	/*Liberamos memoria del DEVICE*/
	cudaFree(mapad);

	/*Liberamos memoria del HOST*/
	free(mapa);
	free(maximo);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();

	return 0;
}




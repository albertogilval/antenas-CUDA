#include <stdio.h>
#include <cuda.h>

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


int reduce(int *c, int *v, int size,dim3 bd,dim3 gd)
{
	int t=size;
	
		while(t!=0){
		
			gpu_reduce<<<gd,bd>>>(c,t);
			cudaDeviceSynchronize();
			if(size%2==0){
				t=t/2;
			}else{
			size -= 1;
			}
		}

		cudaMemcpy(v,c,sizeof(int) * size,cudaMemcpyDeviceToHost);
	return v[0];
}

int main()
{
	int *v, *vd, size, i;
	size = 6;

	if ((v=(int *) calloc(size, sizeof(int)))==NULL){
		printf("error\n");
		exit (-1);
	}

	cudaMalloc( (void**) &vd, sizeof(int) * (int) size);


	int tam = (int) ceil( ((float)(size))/size);
	dim3 bd(128,1);
	dim3 gd(tam,1);

	for(i=0;i<size;i++){
		v[i]=size-i;
	}
	v[5]=50;

	cudaMemcpy(vd,v,sizeof(int) * size,cudaMemcpyHostToDevice);
	
	int max = reduce(vd,v, size,bd,gd);
	printf("%d\n",max);

	cudaFree(vd);
	free(v);
	cudaDeviceReset();
}

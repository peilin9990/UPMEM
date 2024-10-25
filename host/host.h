#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
// User Defined Library

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/gnn_dpu_aggregation"
#endif



void load_data(struct csr* graph,const char* filename){

    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    // 跳过头部信息
    char line_buffer[256];
    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        if (line_buffer[0] != '%') {
            sscanf(line_buffer, "%d %d %d", graph->numRows, graph->numCols, graph->numNonZero);
            break;
        }
    }

    // 初始化CSR格式
    graph->rowPtr = (int *)calloc(graph->numRows + 1, sizeof(int));
    graph->colIdx = (int *)malloc(graph->numNonZero * sizeof(int));
    graph->values = (double *)malloc(graph->numNonZero * sizeof(_Float32));
     memset(graph->values, 0, graph->numNonZero * sizeof(_Float32));
     memset(graph->rowPtr,0,(graph->numRows + 1) *sizeof(int));
     memset(graph->colIdx,0,graph->numNonZero * sizeof(int));

    if(!graph->rowPtr||!graph->colIdx||!graph->values){
		perror("Fail to alloc the memory");
		exit(EXIT_FAILURE);
	}

   
    // 读取矩阵的非零元素
    int currentIndex = 0;
    
	for (int i = 0; i < graph->numNonZero; ++i) {
        uint32_t row, col;
		double value;
        fscanf(file, "%d %d %lf", &col, &row, &value);
        if(row==0) row++;
        graph->rowPtr[row - 1]++; // 更新行指针的计数
        graph->colIdx[currentIndex] = col; // 存储列索引
        currentIndex++;
    }

    // 计算行指针
    for (int i = 1; i <= graph->numRows; ++i) {
        graph->rowPtr[i] += graph->rowPtr[i - 1];
    }

    fclose(file);
}

void printCSR(const struct csr *csr) {
    printf("RowPtr: ");
    for (int i = 0; i <= csr->numRows; i++) {
        printf("%d ", csr->rowPtr[i]);
    }
    printf("\nColIdx: ");
    for (int i = 0; i < csr->numNonZero; i++) {
        printf("%d ", csr->colIdx[i]);
    }
    printf("\nValues: ");
    for (int i = 0; i < csr->numNonZero; i++) {
        printf("%lf ", csr->values[i]);
    }
    printf("\n");
}

void freeCSR(struct csr *graph) {
    free(graph->rowPtr);
    free(graph->colIdx);
    free(graph->values);
}

void freedegree(uint32_t *degree){
    free(degree);
}

matrix feature_init(const struct csr * graph,int32_t feature_dimension){
    
  // 获取图的行数，假设图的行数与特征矩阵的行数相同
    int numRows = graph->numRows; // 假设csr结构体有numRows字段
    matrix features = (matrix)malloc(numRows * sizeof(uint32_t*));

    if (!features) {
        perror("Failed to allocate memory for feature matrix");
        return NULL;
    }

    // 初始化随机数生成器
    srand((unsigned int)time(NULL));

    for (int i = 0; i < numRows; i++) {
        features[i] = (uint32_t*)malloc(feature_dimension * sizeof(uint32_t));
        if (!features[i]) {
            perror("Failed to allocate memory for feature row");
            // 释放已分配的内存
            for (int j = 0; j < i; j++) {
                free(features[j]);
            }
            free(features);
            return NULL;
        }

        // 随机初始化特征值
        for (int j = 0; j < feature_dimension; j++) {
            features[i][j] = rand() % 100; // 生成0到99之间的随机数
        }
    }

    return features;

}

void weight

void adjacency_matrix_normalizaton(struct csr* graph,uint32_t* degree){
for (uint32_t i = 0; i < graph->numRows; i++) {
        for (uint32_t j = graph->rowPtr[i]; j < graph->rowPtr[i + 1]; j++) {
            // 使用度归一化
            if (degree[i] > 0) {
                graph->values[j] = 1.0 / (_Float32)degree[i]; // 归一化
            } else {
                graph->values[j] = 0; // 如果度为0，设置为0
            }
        }
}


}

void host_init_system(int layernum, struct dpu_set_t *rank_pool,int *nr_dpu,int nr_ranks){
if(nr_ranks%layer_num){
    assert(0&&"num of ranks cannot be divided by layernum\n");
}
int ranks_per_layer=nr_ranks/layer_num;

for(int i=0;i<layer_num;i++){
  DPU_ASSERT(dpu_alloc_ranks(ranks_per_layer, NULL, &rank_pool[i])); 
  DPU_ASSERT(dpu_get_nr_dpus(rank_pool[i], &nr_dpu[i]));
  DPU_ASSERT(dpu_load(rank_pool[i],DPU_BINARY, NULL));
}

}

void print_system(int* nr_dpu,int nr_ranks){

    for(int i=0;i<nr_ranks;i++){
        printf("the ith rank has %d dpus\n",nr_dpu[i]);
    }
    
}

void host_transfer_cluster_info (struct dpu_set_t* rank_pool,int32_t layer_num,uint32_t nr_ranks,struct cluster_info_t* cluster_info)
{
        for(int i = 0;i < layer_num;i++)
           for(int j=0;j<nr_ranks/layer_num;j++)
		DPU_ASSERT(dpu_broadcast_to(rank_pool[i*(nr_ranks/layer_num)+j], "cluster_info", 0, cluster_info[i*(nr_ranks/layer_num)+j], sizeof(cluster_info_t), DPU_XFER_DEFAULT));

}


struct csr* split_csr_by_columns(const struct csr *graph, int num_splits) {
   
    int split_cols = graph->numCols / num_splits;
    
    // 创建存储子矩阵的数组
    struct csr *sub_matrices = malloc(num_splits * sizeof(struct csr));
    
    for (int split = 0; split < num_splits; ++split) {
        int start_col = split * split_cols;
        int end_col = (split + 1) * split_cols;
        if (split == num_splits - 1) {
            end_col = graph->numCols; 
        }
        
        
        sub_matrices[split].numRows = graph->numRows;
        sub_matrices[split].numRows = end_col - start_col;
        sub_matrices[split].numNonZero= 0; 
        
        
        sub_matrices[split].values = malloc(graph->numNonZero * sizeof(float));
        sub_matrices[split].colIdx = malloc(graph->numNonZero* sizeof(uint32_t));
        sub_matrices[split].rowPtr = malloc((sub_matrices[split].numRows + 1) * sizeof(uint32_t));
        
        sub_matrices[split].rowPtr[0] = 0;
        int current_nonzeros = 0;
        
      
        for (int row = 0; row < graph->numRows; ++row) {
            for (int idx = graph->rowPtr[row]; idx < graph->rowPtr[row + 1]; ++idx) {
                int col = graph->colIdx[idx];
                if (col >= start_col && col < end_col) {
             
                    sub_matrices[split].values[current_nonzeros] = graph->values[idx];
                    sub_matrices[split].colIdx[current_nonzeros] = col - start_col;
                    current_nonzeros++;
                }
            }
            sub_matrices[split].rowPtr[row + 1] = current_nonzeros;
        }
        
        sub_matrices[split].numNonZero = current_nonzeros;
    }
    
    return sub_matrices;
}



void host_transfer_adjacency_matrix(struct dpu_set_t* rank_pool,const struct csr* graph,int layer_num,int clusters_per_layer,int equal_partition){


    for(int i=0;i<layer_num;i++)
       for(int j=0;j<clusters_per_layer/equal_partition;j++)
          for(int k=0;k<equal_partition;k++)
          {
    uint32_t* row_ptr_buffer;
    uint32_t* col_ptr_buffer;
    uint32_t* value_buffer;


    //为了八字节对齐
    if(graph[j].numRows%2)  row_ptr_buffer=(uint32_t*)malloc((graph[j].numRows+1)*sizeof(uint32_t));
    else row_ptr_buffer=(uint32_t)malloc(graph->numRows*sizeof(uint32_t));
    if(graph[j].numCols%2)  col_ptr_buffer=(uint32_t)malloc((graph[j].numCols+1)*sizeof(uint32_t));
    else col_ptr_buffer=(uint32_t)malloc((graph->numCols)*sizeof(uint32_t));
    if(graph[j].numNonZero%2) value_buffer=(float*)malloc((graph[j].numNonZero+1)*sizeof(float));
    else value_buffer=(float*)malloc((graph->numNonZero)*sizeof(float));
    memcpy(row_ptr_buffer,graph[j].rowPtr,graph[j].numRows);
    memcpy(col_ptr_buffer,graph[j].colIdx,graph[j].numCols);
    memcpy(value_buffer,graph[j].values,graph[j].numNonZero);
    DPU_ASSERT(dpu_broadcast_to(rank_pool[i*clusters_per_layer+j*equal_partition+k], "row_ptr_mram", 0, row_ptr_buffer, sizeof(uint32_t)*(graph[j].numRows+graph[j].numRows%2), DPU_XFER_DEFAULT));
     DPU_ASSERT(dpu_broadcast_to(rank_pool[i*clusters_per_layer+j*equal_partition+k], "col_idx_mram", 0, col_ptr_buffer, sizeof(uint32_t)*(graph[j].numCols+graph[j].numCols%2), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(rank_pool[i*clusters_per_layer+j*equal_partition+k], "value_mram", 0, value_buffer, sizeof(uint32_t)*(graph[j].numNonZero+graph[j].numNonZero%2), DPU_XFER_DEFAULT));
          }

}






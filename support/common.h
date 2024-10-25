#ifndef _COMMON_H_
#define _COMMON_H_


#ifndef PRINT
#define PRINT 0
#endif

// Structures used by both the host and the dpu to communicate information 
typedef struct {
    uint32_t n_size;
    uint32_t n_size_pad;
    uint32_t nr_rows;
    uint32_t max_rows;
} dpu_arguments_t;

// Specific information for each DPU
struct dpu_info_t {
  uint32_t dpu_id;
  uint32_t max_rows_per_dpu;
  uint32_t rows_per_dpu_pad;
  uint32_t start_row;
};



struct cluster_info_t {
  uint32_t cluster_id;
  uint32_t feature_dimension;
  uint32_t feature_row_num;
  uint32_t adjacency_rows;
};



struct dpu_info_t *dpu_info;

struct csr {
 uint32_t * rowPtr;
 uint32_t * colIdx;
 float *values;
 uint32_t numRows;
 uint32_t numCols;
 uint32_t numNonZero;
}



#define MAX_RANK_NUM 32
#define NUM_LAYERS 1
#define CLUSTERS_PER_RANK 1
#define CORES_PER_RANK 1
#define EQUAL_PARTITION 1
#define max(x, y) (x > y ? x : y)
#define min(x, y) (x < y ? x : y)

// Transfer size between MRAM and WRAM
#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

//Dataset path
#define FileName "./Dubcova2.txt"



// Data type
#define T int32_t
#define matrix uint32_t** 

#define PRINT 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#endif

/**
 * host.c
 * GNN Host Application Source File
 *
 */

#include<host.h>
struct dpu_set_t rank_pool[MAX_RANK_NUM];
int32_t nr_dpu_per_rank[MAX_RANK_NUM];
struct cluster_info_t * cluster_info;
struct dpu_info_t * dpu_info;
Timer timer;
int32_t feature_dimension;
int32_t layer_num;
int32_t warm_ups;
int32_t batch_size;
int32_t reps;
int32_t equal_partition;

int main(int argc, char **argv) {

	struct Params p = input_params(argc, argv);

	struct dpu_set_t dpu_set, dpu;
	uint32_t nr_of_ranks=p.n_ranks;
	uint32_t clusters_per_rank=CLUSTERS_PER_RANK;
	uint32_t cores_per_rank=CORES_PER_RANK;
    
    struct csr graph;
	// Allocate DPUs and load binary
	// DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
	// DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
	// DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

	//加载数据并以csr的形式存储在graph里面
    feature_dimension=p.feature_dimension;
    layer_num=p.layer_num==0: NUM_LAYERS:p.layer_num;
    warm_ups=p.n_warmup;
	reps=p.n_reps;
    start(&timer,0,0);
	start(&timer,0,1);
    load_data(&graph,FileName);
	stop(&timer,0);
	printf("time to load data is ");
	print(&timer,0,1);

	//初始化feature向量
    matrix feature_matrix=feature_init(&graph,feature_dimension);
	printf("feature initialized \n");

	//计算度矩阵并归一化邻接矩阵

	uint32_t *degree = (uint32_t *)calloc(graph.numRows, sizeof(uint32_t));
    for (uint32_t i = 0; i < graph.numRows; i++) {
        for (uint32_t j = graph.rowPtr[i]; j < graph.rowPtr[i + 1]; j++) {
            degree[i] += graph.values[j]; // 假设values数组存储的是邻接矩阵的非零值
        }
    }
    
    adjacency_matrix_normalizaton(&graph,degree);

    host_init_system(layer_num,rank_pool,nr_dpu_per_rank,nr_of_ranks);

	#if PRINT
	print_system(nr_dpu_per_rank,nr_of_ranks);
	#endif

	equal_partition=p.equal_partition==0? EQUAL_PARTITION:p.equal_partition;
	if(feature_dimension%equal_partition)
		assert(0&&"feature dimension cannot be diveded by partition \n ");

	for(int i=0;i<layer_num;i++){
		 cluster_info=(struct cluster_info_t *)malloc((nr_of_ranks)*sizeof(struct cluster_info_t));
	  for(int j=0;j<nr_of_ranks/layer_num;j++)
	  {
		//每个cluster 对应一个rank
		cluster_info[i*(nr_of_ranks/layer_num)+j].cluster_id=i<<16+j;
		
		cluster_info[i*(nr_of_ranks/layer_num)+j].feature_dimension=feature_dimension/equal_partition;

		cluster_info[i*(nr_of_ranks/layer_num)+j].feature_row_num=graph.numRows/((nr_of_ranks/layer_num)/equal_partition);

		cluster_info[i*(nr_of_ranks/layer_num)+j].adjacency_rows=graph.numRows;
	  }
	}



//   传输cluster信息
    host_transfer_cluster_info(rank_pool,layer_num,nr_of_ranks,cluster_info);

// 传输feature_matrix


uint32_t cluster_idx=0;
          for(int i=0;i<layer_num;i++){
			uint32_t cluster_per_layer=nr_of_ranks/layer_num;
			for(int j=0;j<(nr_of_ranks/layer_num)/equal_partition;j++)
				for(int k=0;k<equal_partition;k++)
				{
						uint32_t row_num=graph.numRows/((nr_of_ranks/layer_num)/equal_partition);
						if(row_num%2){
							row_num++;
						}			
						uint32_t row_per_dpu=graph.numRows/((nr_of_ranks/layer_num)/equal_partition);
						uint32_t col_per_dpu=feature_dimension/equal_partition;

						uint32_t * transfer_buffer;
						transfer_buffer=(uint32_t*)malloc(row_num*feature_dimension/equal_partition*sizeof(uint32_t));
						memcpy(transfer_buffer,feature_matrix[j*(graph.numRows/((nr_of_ranks/layer_num)/equal_partition))]+k*(feature_dimension/equal_partition),row_per_dpu*col_per_dpu*sizeof(uint32_t));
						DPU_ASSERT(dpu_broadcast_to(rank_pool[i*cluster_per_layer+j*equal_partition+k], "feature_matrix_mram", 0, transfer_buffer, sizeof(uint32_t)*row_per_dpu*col_per_dpu, DPU_XFER_DEFAULT));
						free(transfer_buffer);
				}
		}

		  
//为每个dpu分配vertice
  struct dpu_info_t* dpu_info;
  dpu_info=(struct dpu_info_t*)malloc(nr_dpu_per_rank[i]*sizeof(struct dpu_info_t));
     for(int i=0;i<nr_of_ranks;i++)	{
		int j=0;
		uint32_t previous_rows=0;
		uint32_t max_row_num=0;
		if(graph.numRows%nr_dpu_per_rank[i])  max_row_num=graph.numRows/nr_dpu_per_rank[i]+1;
		else max_row_num=graph.numRows/nr_dpu_per_rank[i];

       DPU_FOREACH(rank_pool[i], dpu, j){
		dpu_info[j].dpu_id=i<<16+j;
		dpu_info[j].start_row=previous_rows;

       if(j<graph.numRows%nr_dpu_per_rank[i]) 
	    dpu_info[j].rows_per_dpu=max_row_num
	   else
	    dpu_info[j].rows_per_dpu=graph.numRows/nr_dpu_per_rank[i];
		
		previous_rows+=dpu_info[j].rows_per_dpu;
		dpu_info[j].max_rows_per_dpu=max_row_num;
        DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_info+j));
	 }
      DPU_ASSERT(dpu_push_xfer(rank_pool[i], DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(struct dpu_info_t), DPU_XFER_DEFAULT));
	 }


   //按列切分adjaceney矩阵
   struct csr * sub_graph;
   uint32_t clusters_per_layer=nr_of_ranks/layer_num;
   int split_num=clusters_per_layer/equal_partition;
   sub_graph=split_csr_by_columns(&graph,split_num);

   //分别为每一个cluster传输adjacency的子矩阵
    host_transfer_adjacency_matrix(rank_pool,sub_graph,layer_num,clusters_per_layer,equal_partition);

	//每一个layer 包含一个aggregation 一个线性层 和一个激活函数
	//定义变量保存每一层的输出
	 uint32_t cluster_output[graph.numRow][feature_dimension/equal_partition];
	 uint32_t aggregation_output[graph.numRow][feature_dimension];
      start(&timer,2,0);
	  start(&timer,2,1);
		for(int lay = 1; lay < NUM_LAYERS; lay++){
			
        for(int i=0;i<clusters_per_layer;i++){
       DPU_ASSERT(dpu_launch(rank_pool[lay*clusters_per_layer+i], DPU_SYNCHRONOUS));

		 for(int i=0;i<clusters_per_layer;i++){
			DPU_FOREACH(rank_pool[lay*clusters_per_layer], dpu, j) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, cluster_output[dpu_info[j]]));
			}
			
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "dpu_output_mram", dpu_info[j].start_row*feature_dimension/equal_partition*sizeof(uint32_t),dpu_info[j].rows_per_dpu*feature_dimension/equal_partition*sizeof(uint32_t), DPU_XFER_DEFAULT));
			//计算该cluster是第几组
			int group_id=i%equal_partition;
			//将整个cluster的输出copy到aggregation_output中
			for(int m=0;m<graph.numRows;m++)
			 for(int n=feature_dimension/equal_partition*group_id;n<feature_dimension/equal_partition*(group_id+1);n++)
			    aggregation_output[m][n]+=cluster_output[m][n%(feature_dimension/equal_partition)];
		 }
			
    // 在cpu上面计算combination
	

		// Retrieve results
		if (rep >= p.n_warmup)
			start(&timer, 3, rep - p.n_warmup);
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + i * max_rows_per_dpu));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
		if(rep >= p.n_warmup)
			stop(&timer, 3);
	}

#if ENERGY
	double acc_energy, avg_energy, acc_time, avg_time;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_ACCUMULATE, &acc_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &avg_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_ACCUMULATE, &acc_time));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_AVERAGE, &avg_time));
#endif

	// Print timing results
	printf("CPU Version Time (ms): ");
	print(&timer, 0, 1);
	printf("CPU-DPU Time (ms): ");
	print(&timer, 1, p.n_reps);
	printf("DPU Kernel Time (ms): ");
	print(&timer, 2, p.n_reps);
	printf("Inter-DPU Time (ms): ");
	print(&timer, 4, p.n_reps);
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif
	printf("\n\n");

	// Check output
	bool status = true;
	unsigned int n, j;
	i = 0;
	for (n = 0; n < nr_of_dpus; n++) {
		for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
			if(C[i] != C_dpu[n * max_rows_per_dpu + j]) {
				status = false;
#if PRINT
				printf("%d: %d -- %d\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
#endif
			}
			i++;
		}
	}
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
	}

	// Deallocation
	for(i = 0; i < NUM_LAYERS; i++)
		free(A[i]);
	free(A);
	free(B);
	free(C);
	free(C_dpu);
	DPU_ASSERT(dpu_free(dpu_set));

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

	return status ? 0 : -1;
}



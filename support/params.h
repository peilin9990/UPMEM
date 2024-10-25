#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "common.h"

typedef struct Params {
    unsigned int  layer_num;
    unsigned int  feature_dimension;
    unsigned int batch_size;
    unsigned int  n_warmup;
    unsigned int  n_reps;
    unsigned int n_ranks;
    unsigned int equal_partition ;
}Params;

static void usage() {
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\nGeneral options:"
            "\n    -h        help"
            "\n    -l <L>    # of layernum"
            "\n    -f <F>    # of feature_dimension "
            "\n    -w <W>    # of warmup "
            "\n    -r <R>    # of reps "
            "\n    -k <K>    # of ranks "
            "\n    -p <P>    # of equal partition "
            "\n");
}

struct Params input_params(int argc, char **argv) {
    struct Params p;
    p.layer_num        = 1;
    p.feature_dimension    = 1024;
    p.batch_size       = 1;
    p.n_warmup      = 0;
    p.n_reps        = 1;
    p.n_ranks       =MAX_RANK_NUM;

    int opt;
    while((opt = getopt(argc, argv, "hm:n:w:e:")) >= 0) {
        switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'l': p.layer_num        = atoi(optarg); break;
            case 'f': p.feature_dimension        = atoi(optarg); break;
            case 'w': p.n_warmup      = atoi(optarg); break;
            case 'r': p.n_reps        = atoi(optarg); break;
            case 'k': p.n_ranks       = atoi(optarg); break;
            case 'p': p.equal_partition   =atoi(optarg); break;
            default:
                      fprintf(stderr, "\nUnrecognized option!\n");
                      usage();
                      exit(0);
        }
    }
    assert(NR_DPUS > 0 && "Invalid # of dpus!");

    return p;
}
#endif

/**************************************************************************
Copyright:
Author: Peng Baoyun
Data:   2016-07-30
Description:Regression Random  Forest implementation on Intel Xeon Phi(MIC).
***************************************************************************/

#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <sys/time.h>
#include <ctime>
#include <errno.h>
#include <cassert>
#include <omp.h>
#include <getopt.h>
#include <stdio.h>  
#include <unistd.h>

using namespace std;

#define LOG_ERROR_(message) fprintf(stderr, "%s:%d:%s():  %s", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_WARNING_(message) fprintf(stderr, "%s:%d:%s (): %s", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_NOTICE_(message) fprintf(stderr , "%s",  message);
#define sign(val) val>EPSI?1:(val<(-EPSI)?-1:0)
#define square(x) (x*x)

#define BUFFER_LENGTH	65536
#define MAX_CHILDS_NUM 	128
#define MAX_NODES_NUM  	512
#define MAX_TREE_NUM 	200

/* configuration information for constructing forest */
typedef struct {   
	int data_num;	 /* samples number. */
	int max_feature; /* sample's features number */
	int tree_num;	 /* trees number in forest */
	int depth;		 /* maximum depth in trees */
	int min_children;/* minimum samples number in leaf nodes */
	float bootstrap; /* sampling ratio for samples */
	int nthread;	 /* maximum parallel threads number */
} ForestConfig;

/* tree's node in forest */
typedef struct {    
	float value;	/* average value of samples in current node. */
	float splitval;	/* the splitting value in current node. */
	int ind;		/* the splitting feature id in current node. */
	int ch[2];		/* samples number of left and right child. */
	float sum_y;	/* sum of samples in current node. */
	float sum_sqr_y;/* square sum of samples in current node. */
} TNode;

/* the pair data structure, for storing the data by column(features) instead of by row(samples) */
typedef struct {
	float val;	/* the feature's value */
	int   pos;	/* the feature's position */
} FPpair;

/* the splitting basic information */
struct  SplitInfo{
	int bind;		/* the feature's index of the best split pos */
	float bsplit;	/* the best splitting's value' */
	int cnt[2];		/* the left and right child nodes samples number after splitting' */
	float sum_y[2]; /* sum of samples value in left and right child nodes */
	float sum_sqr_y[2];  /* square sum of samples value in left and right child nodes */
	float err;		/* the splitting gain by current splitting */
	void update(const SplitInfo &sinfo) {
		bind = sinfo.bind;
		bsplit = sinfo.bsplit;
		cnt[0] = sinfo.cnt[0]; cnt[1] =  sinfo.cnt[1];
		sum_y[0] = sinfo.sum_y[0]; sum_y[1] =  sinfo.sum_y[1];
		sum_sqr_y[0] = sinfo.sum_sqr_y[0];  sum_sqr_y[1] = sinfo.sum_sqr_y[1];
		err = sinfo.err;
	}
};

#pragma offload_attribute(push,target(mic))

/* split information of needing splitting's' nodes in the same depth */
struct QNode{
public:
	int pid;		/* node's father node id' */
	int cnt;		/* current samples number in this node */
	float err;		/* current split gain in this node */
	QNode(){pid = cnt = 0; err = 0.0f;}
	QNode(const int &pid_, const int &cnt_, const  float &err_) {
		pid = pid_;
		cnt = cnt_;
		err = err_;
	}
} ;

static TNode forestNodes[MAX_NODES_NUM * MAX_TREE_NUM];	/* forest's nodes array */
static int   treeNodesNum[MAX_NODES_NUM];	/* number of tree's nodes in forest' */
float 		*dataVec;		/* the samples and features value */
float		*yVec;			/* the target value of all samples */


/* the comparable function for sorting the feature pairs */
inline bool mySmaller(const FPpair &a, const FPpair &b)
{
	return a.val < b.val;
}

/************************************************* 
Description:   find the best splitting positions given the feature(feas)
				array for all the sample depth's nodes
Calls:          buildDecisionTree()
Input:     
	fppair: FPpair[] : the feature array need to find the best splitting positions 
	feas_num: int  : the element number in feas array    
	samples_num:int : the samples number in fppair
	feature_id: int : the index of feature needing to find splitting pos
	positions: int[]: the samples positions array recording the index of nodes
	sqr_y_list: float[]: the samples square value in feas array
	q: QNode[] : the nodes needing to split in the same depth 
	samples_ids: int[] : the index of samples in dataVec 
Output:       
	bsplit_info: SplitInfo[] : the split information 
	 
Return:         NULL
*************************************************/ 
inline void find_splits(FPpair *fppairs, int samples_num, int feature_id, int *positions, 
		float *sqr_y_list, QNode *q, int *samples_ids, SplitInfo *splits_info, TNode *treeNodes, int sampled_fea_num)
{
	/* the samples sum value in left child nodes if splitting */
	float *left_childs_sum = (float*) malloc(sizeof(float) * MAX_CHILDS_NUM);
	/* the samples number in left child nodes if splitting */	
	int	  *left_childs_num = (int *) malloc(sizeof(int) * MAX_CHILDS_NUM);	

	/* traversal all values in feas to find the best split pos */
	for(int i=0; i<samples_num; i++)
	{
		/* the index in dataVec of current sample  */
		int sid = samples_ids[fppairs[i].pos];
		/* the index of node in current depth containing current sample */
		int cid = positions[sid];
		/* the index of node's father node */
		int pid = q[cid].pid;
		
		left_childs_sum[cid] += yVec[sid];
		left_childs_num[cid]++;
		/* computing the split gain of current split pos, split_gain = sum^2/N - sum1^2/L - sum2^2/R */
		float tempGain = q[cid].err - square(left_childs_sum[cid]) / left_childs_num[cid] + 
			square(treeNodes[pid].sum_y - left_childs_sum[cid]) / (q[cid].cnt - left_childs_num[cid]);
		
		SplitInfo &bsplit_info = splits_info[cid * sampled_fea_num +feature_id];
		
		if(tempGain < bsplit_info.err)
		{
				bsplit_info.err = tempGain;
				bsplit_info.bind = feature_id;
				bsplit_info.bsplit = fppairs[feature_id*samples_num + i].val;
				bsplit_info.cnt[0] = left_childs_num[cid];	
				bsplit_info.cnt[1] = q[cid].cnt - left_childs_num[cid];
				bsplit_info.sum_y[0] = left_childs_sum[cid];
				bsplit_info.sum_y[1] = treeNodes[pid].sum_y - left_childs_sum[cid];
				bsplit_info.sum_sqr_y[0] = left_childs_sum[cid];
				bsplit_info.sum_sqr_y[1] = treeNodes[pid].sum_y - left_childs_sum[cid];
		}
	}
}


/************************************************* 
Description:   constructing regression tree 
Input:           
	samples_ids: int[] : the samples index array used for constructing this tree
	samples_num: int  :the samples number 
	config: ForestConfig : the basic information
Output: 
	treeNodes: TNode[] : the nodes array in this tree after constructing
	nodes_num: int 		: the nodes number in this tree after constructing
Return:         NULL
*************************************************/ 
void buildDecisionTree(int *samples_ids, int samples_num, ForestConfig &config,
		 TNode *treeNodes, int &nodes_num)
{
	/* the square values of all samples in the same depth */
 	float *sqr_y_list = (float*) malloc(sizeof(float) * config.data_num);	

	/* the target average value and square value of all samples */
    float sum_y = 0.0f, sum_sqr_y = 0.0f;	
	
	/* the nodes's positions of samples in the same depth */
	int   *positions  = (int  *) malloc(sizeof(int) * MAX_CHILDS_NUM);		
   
    int   currLevel_childs_num = 0;	/* the nodes number in the sample tree's depth */
	int   sampled_fea_num = sqrt(config.max_feature);		/* features number when splitting at tree's node */
	
	/* the nodes needing to splitting in the sample depth */
	QNode *q = (QNode *) malloc(sizeof(QNode) *  MAX_CHILDS_NUM);
	
	/* fppairs :storing the samples data by column(features) instead of by row(samples) */
	FPpair *fppairs = (FPpair*) malloc( sizeof(FPpair) * config.data_num * config.max_feature);
	
	/* splits_info: storing the splitting informations for all features and all nodes */
	SplitInfo *splits_info = (SplitInfo *) malloc(sizeof(SplitInfo) * MAX_CHILDS_NUM * sampled_fea_num);
	
	/* the best split pos for all nodes in the same depth */
	SplitInfo *best_split_info = (SplitInfo *) malloc(sizeof(SplitInfo) * MAX_CHILDS_NUM * sampled_fea_num);

	/* storing the data by column(features) in fppairs instead of by row(samples) */
	for (int i = 0; i < samples_num; i++)  
	{
		int sid = samples_ids[i];
		int fid = sid * config.max_feature;
		sqr_y_list[i] = square (yVec[sid]);
		sum_y += yVec[sid];
		sum_sqr_y += sqr_y_list[i];                
		positions[i] = 0;
		for(int j = 0; j < config.max_feature; j++, fid++)
		{
			fppairs[j * config.data_num + i].pos = sid;
			fppairs[j * config.data_num + i].val = dataVec[ fid ];
		}
	}

	/* sorting the features value by ascending */
	for (int i = 0; i < config.max_feature; i++)
	    sort((fppairs + i*config.data_num), (fppairs + (i+1) * config.data_num), mySmaller);           
    
	/* adding the root node in this tree */
	nodes_num = 0;
	TNode &node = treeNodes[nodes_num++];
	node.ind = -1;	
	node.value = sum_y / (samples_num ? samples_num : 1);
	node.sum_y = sum_y;
	node.sum_sqr_y = sum_sqr_y;

	/* add the first node needing to splitting */
    q[currLevel_childs_num++] = (QNode(0,  samples_num, sum_sqr_y- sum_y*sum_y/samples_num));  

	/* the samples sum value in left child nodes if splitting */
	float *left_childs_sum = (float*) malloc(sizeof(float) * MAX_CHILDS_NUM);
	/* the samples number in left child nodes if splitting */	
	int	  *left_childs_num = (int *) malloc(sizeof(int) * MAX_CHILDS_NUM);	
	
    /* constructing the tree by breadth-first-search */
	for (int dep = 0; dep < config.depth;  dep++) {
		if (currLevel_childs_num == 0)  break; /* stop constructing if no nodes need to split */
		
		/* initial the split information */
		for (int i = 0; i <  currLevel_childs_num; i++) {               
			for (int j = 0; j <  sampled_fea_num; j++) {                    
				splits_info[i*sampled_fea_num].err = q[j].err;
				splits_info[i*sampled_fea_num].bind = -1;
			}
		}

		/* find the best splitting positions given the feature(feas) array for current depth */
		for (int findex = 0; findex <sampled_fea_num; findex++)
		{
			/* randomly select feature to find split position */
			int feature_id = (int) rand() % config.max_feature;
			find_splits( fppairs, samples_num, feature_id, *positions, 
					sqr_y_list, q, samples_ids, splits_info, treeNodes, sampled_fea_num)	
		}

		/* merge split positions on all features to a best splitting */
		for (int i = 0; i < currLevel_childs_num; i++) {
			SplitInfo &spinfo =  best_split_info[i];
			spinfo.bind = -1;
			for (int j = 0; j <  sampled_fea_num; j++)
				if (tinfos[j][i].spinfo.bind >= 0 && (spinfo.bind < 0 
					|| spinfo.err >  tinfos[j][i].spinfo.err))
					spinfo.update(tinfos[j][i].spinfo);
		}
		/* split nodes to further depth */
		update_queue(tree, nodes_num);

	}
}

/************************************************* 
Description:   building forest
Input:        
	dataVec: float[]: the samples and features data   
	yVec   : float[]: the target value of all samples
	config: ForestConfig : the configuration for constructing forest
Output: 
	regression forest 
Return:         NULL 
*************************************************/ 
void buildForest(float *dataVec, float *yVec, ForestConfig &config)
{        
	srand((unsigned)time(NULL));
	int *samples_ids = (int *)malloc(sizeof(int) * config.data_num);
	for (int i = 0; i < config.tree_num; i++)
	{
		/* randomly sampling the data to construct tree */
		for(int j = 0; j < config.data_num; j++)
			samples_ids[j] = (int)rand() % config.data_num;
		buildDecisionTree(samples_ids, config.data_num, config, &(forestNodes[i*MAX_NODES_NUM]), treeNodesNum[i]);
	}
}

#pragma offload_attribute(pop)

/************************************************* 
Description:   get the maximum features number for a sample(string format)
Calls:          read_conf_file()
Input:          line:String, the sample
Return:         the maximum features number 
*************************************************/ 
int max_feature_label(std::string line)
{
	int start = 0;
	int fid;
	int max_fid = 0;
	int len = line.length();

	for(int i = 0; i < len; i++) {
		if(line[i] == ' ') {
			start = i+1;
		}
		else if(line[i] == ':') {
			if(sscanf(line.substr(start, i  - start).c_str(), "%d", &fid) == 1) {
				if(max_fid < fid) {
					max_fid = fid;
				}
			}	
		}
	} 
	return max_fid;
}

/************************************************* 
Description:   splitting the string line by separator
Calls:          read_conf_file()
Input:          line:String, the sample
Output:			the string array after splitting
Return:         the elements number in items 
*************************************************/ 
int splitline(std::string line, std::string items[], int  items_num, const char separator)
{
	if(items == NULL || items_num <= 0) {
		return -1;
	}
	int n = line.length();
	int j = 0;
	int start = 0;

	for(int i = 0; i < n; i++) {
		if(line[i] == separator) {
			if(j < items_num && start < n) {
				items[j++] = line.substr (start, i-start);
				start = ++i;
			}
		}
	}
	if(j < items_num && start < n) {
		items[j++] = line.substr(start, n-start);
	}
	return j;
}

bool has_colon(std::string item)
{
	for(int i = 0; i < (int)item.size(); i++)
	{
		if(item[i] == ':')
			return true;
	}
	return false;
}

/************************************************* 
Description:   print help information
*************************************************/ 
int print_usage (FILE* stream, char* program_name)
{
	fprintf (stream, "Usage: %s options [ input file  ... ]\n", program_name);
	fprintf (stream, " -h --help.\n"
		" -r --sample_feature_ratio Feature ratio  learning rate.\n"
		" -t --tree_num GBDT #trees in the model \n"
		" -s --shrink : learning rate\n"
		" -n --min_node_size : stop tree growing,  minimum samples in leaf node.\n"
		" -d --max_depth stop tree growing,  maximum depth of the tree.\n"
		" -m --model_out .\n"
		" -f --train_file .\n"
		);

	return 0;
}


/************************************************* 
Description:   getting the samples data and basic information from file
*************************************************/ 
int read_conf_file(ForestConfig& config, int argc, char*  argv[])
{
	if(argv == NULL)	{
		LOG_ERROR_("Parameter error.");
		return -1;
	}

	int ch;
	double random_feature_ratio = -1;
	char message[BUFFER_LENGTH];

	const char* short_options = "h:r:t:n:d:m:f:q:";
	char filename[BUFFER_LENGTH];
	const struct option long_options[]={
		{"help", 0, NULL, 'h'},
		{"sample_feature_ratio", 1, NULL, 'r'},
		{"tree_num", 1, NULL, 't'},
		{"thread_num", 1, NULL, 'q'},
		{"min_node_size", 1, NULL, 'n'},
		{"max_depth", 1, NULL, 'd'},
		{"model_out", 1, NULL, 'm'},
		{"train_file", 1, NULL, 'f'},
		{NULL, 0, NULL, 0}
	};

	while((ch = getopt_long (argc, argv,  short_options, long_options, NULL)) != -1)
	{
		switch(ch)
		{
		case 'h':
			if(argc == 2)
			{
				print_usage(stderr, argv [0]);
				return 1;
			}	
			else
				return -1;
		case 'q':
			if(sscanf(optarg, "%d",  &config.nthread) != 1)
			{
				if(config.nthread < 1)
				{
					LOG_ERROR_("Get  thread number error.");
					return -1;
				}
				
			}
			printf("nthread %d\n", config.nthread);
			break;
		case 'r':
			if(sscanf(optarg, "%f",  &config.bootstrap) != 1)
			{
				printf("error 2\n");
				LOG_ERROR_("Get  random_feature_ratio config error.");
				return -1;
			}
			printf("bootstrap %f\n", config.bootstrap);
			break;
		case 't':
			if(sscanf(optarg, "%d",  &config.tree_num) != 1)
			{
				LOG_ERROR_("Get tree_num  config error.");
				return -1;
			}
			printf("tree number %d\n", config.tree_num);
			break;
		case 'n':
			if(sscanf(optarg, "%d",  &config.min_children) != 1)
			{
				LOG_ERROR_("Get  min_node_size config error.");
				return -1;
			}
			printf("min children  %d\n", config.min_children);
			break;
		case 'd':
			if(sscanf(optarg, "%d",  &config.depth) != 1)
			{
				LOG_ERROR_("Get max_depth  config error.");
				return -1;
			}
			printf("max depth%d\n", config.depth);
			break;
		case 'f':
			if(strlen(optarg) <=  BUFFER_LENGTH)
				strncpy((char*) (filename), optarg, BUFFER_LENGTH);
			else
			{
				LOG_ERROR_("Get  train_filename config error.");
				return -1;
			}
			break;
		case 'm' :
			//if(strlen(optarg) <=  BUFFER_LENGTH)
			//strncpy((char*) (filename), optarg, BUFFER_LENGTH);
			break;
		case '?':
			printf("error  1\n");
			print_usage(stderr, argv[0]);
			return -1;
		default:
			printf("error  3\n");
			print_usage(stderr, argv[0]);
			return -1;
		}
	}
	ifstream fin(filename);
	if(fin)
	{
		string line;
		config.data_num = 0;
		config.max_feature = 0;
		int temp;
		while(getline(fin, line))
		{
			temp = max_feature_label(line);
			if(config.max_feature < temp)
				config.max_feature =  temp;
			config.data_num++;
		}
		fin.close();
	}else {
		printf("read file error!\n");
		return -1;
	}

	printf("data number %d and max feature %d number!\n", config.data_num, config.max_feature);

	//dataVec.resize (config.data_num*config.max_feature);
	dataVec = (float*)malloc(sizeof(float) *  config.max_feature * config.data_num);
	//yVec.resize(config.data_num);
	yVec = (float*)malloc(sizeof(float) *  config.data_num);
	string* items = new string[config.max_feature+1]; 
	string line;
	int x_read, count,cnt = 0, fid;
	double value;
	printf("start reading data \n");
	ifstream fin1(filename);
	while (getline(fin1, line) != NULL) {
		count = splitline(line, items,  config.max_feature+5, ' '); 
		sscanf(items[0].c_str(),"%1f",&value);
		//printf("value %f \n", value);
		//the first column is y
		yVec[cnt] = value; 
		int temp = cnt * config.max_feature;
		for(int i = 1; i < count; i++) {
			if(has_colon(items[i]))	{
				//featureid1:value1  featureid2:value2 ... density matrix x
				x_read = sscanf(items [i].c_str(),"%d:%lf", &fid, &value); 
				dataVec[temp+fid] =  value;
			}
		}
		cnt++;
		if (cnt >= config.data_num)	{ 
			break;
		}
	}
	
	fin1.close();
	printf("reading data over!\n");
	
	delete[] items;
	return 0;
}

int main(int argc, char* argv[])
{
	ForestConfig config;
	read_conf_file(config, argc, argv);
	
	double start_time, end_time;
	
	#pragma offload target(mic : 0)\
    in(dataVec:length(config.data_num*config.max_feature) alloc_if(1))\
    in(yVec:length(config.data_num) alloc_if(1))
	//out(forestNodes:length(config.tree_num* MAX_NODES_NUM))\
	//out(treeNodesNum:length(config.tree_num))
	{
		buildForest(dataVec, yVec, config);
	}

    printf("forest building feature parallel executing time: %f\n", end_time - start_time);
	
}
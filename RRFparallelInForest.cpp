/***************************************************************************
Copyright:
Author: Peng Baoyun
Date:   2016-07-30
Description:Regression Random  Forest implementation on Intel Xeon Phi(MIC).

	Regression Random Forest, Parallel Implementation, each thread response for
construct a tree, and using oob.
***************************************************************************/

#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <stdio.h>  

using namespace std;

#define sign(val) val>EPSI?1:(val<(-EPSI)?-1:0)
#define square(x) (x*x)

#define BUFFER_LENGTH	65536
#define MAX_CHILDS_NUM 	128
#define MAX_NODES_NUM  	512
#define MAX_TREE_NUM 	200
#define MAX_ERROR		1000000000

//#pragma offload_attribute(push,target(mic))

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



static TNode forestNodes[MAX_NODES_NUM * MAX_TREE_NUM];	/* forest's nodes array */
static int   treeNodesNum[MAX_NODES_NUM];	/* number of tree's nodes in forest' */
float 		*dataVec;		/* the samples and features value */
float		*yVec;			/* the target value of all samples */
ForestConfig config;        /* the configuration of the forest */


class DecisionTree
{
private:
	/* the pair data structure, for storing the data by column(features) instead of by row(samples) */
	struct FPpair{
		float val;	/* the feature's value */
		int   pos;	/* the feature's position */
	} ;

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
	};

public:
	
 	float *sqr_y_list;	/* the square values of all samples in the same depth */
    float sum_y, sum_sqr_y;	/* the target average value and square value of all samples */
	float *left_childs_sum;/* the samples sum value in left child nodes if splitting */
	float *left_childs_sqrsum;/* the samples square sum value in left child nodes if splitting */
	
	int	  *left_childs_num;	/* the samples number in left child nodes if splitting */
	int   *positions;		/* the nodes's positions of samples in the same depth */
    int   currLevel_childs_num;	/* the nodes number in the sample tree's depth */
	int   sampled_fea_num;/* features number when splitting at tree's node */
	int	  nodes_num; /* the number of tree nodes */

	TNode *tree;	/* the array of nodes */
	
	QNode *q;	/* the nodes needing to splitting in the sample depth */
	
	FPpair *fppairs; /* fppairs :storing the samples data by column(features) instead of by row(samples) */
	
	/* splits_info: storing the splitting informations for all sampling features and all nodes */
	SplitInfo *splits_info;
	
	SplitInfo *best_split_info;/* the best split pos for all nodes in the same depth */
	


	/* initial the value and malloc the memory */
	void initData(ForestConfig config, TNode *tree_)
	{
		sum_y = 0.0f; 
		sum_sqr_y = 0.0f;	
		currLevel_childs_num = 0;
		sampled_fea_num = sqrt((float)config.max_feature);

		sqr_y_list = (float*) malloc(sizeof(float) * config.data_num);

		positions  = (int  *) malloc(sizeof(int) * MAX_CHILDS_NUM);		
		
		q = (QNode *) malloc(sizeof(QNode) *  MAX_CHILDS_NUM);
		
		fppairs = (FPpair*) malloc( sizeof(FPpair) * samples_num * config.max_feature);
		
		splits_info = (SplitInfo *) malloc(sizeof(SplitInfo) * MAX_CHILDS_NUM * sampled_fea_num);

		best_split_info = (SplitInfo *) malloc(sizeof(SplitInfo) * MAX_CHILDS_NUM);

		left_childs_sum = (float*) malloc(sizeof(float) * curr_childs_num);

		left_childs_num = (int *) malloc(sizeof(int) * curr_childs_num);	

		left_childs_sqrsum = (float*) malloc(sizeof(float) * curr_childs_num);

		tree = tree_;
	}
	
	/* free the memory */
	void freeData()
	{
		free(left_childs_sum);
		free(left_childs_num);
		free(left_childs_sqrsum);
		free(sqr_y_list);
		free(positions);
		free(fppairs);
		free(splits_info);
		free(samples_ids);
		free(q);
	}

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
		splits_info: SplitInfo : the temporary split information 
								for sampled features of all nodes in the same depth
		treeNodes: TNode*  :  the root node of current tree 
		sampled_fea_num: int : the number of sampling features used to find splitting pos 
		curr_childs_num: int: the number of childs nodes in current depth
		findex:   int       the index of features now splitting
	Output:       
		bsplit_info: SplitInfo[] : the split information 
		
	Return:         NULL
	*************************************************/ 
	void find_splits(int feature_id, QNode *q, int findex)
	{
		for(int i=0; i<curr_childs_num; i++)
		{
			left_childs_sum[i] = 0.0f;
			left_childs_num[i] = 0.0f;
			left_childs_sqrsum[i] = 0.0f;
		}

		/* traversal all values in feas to find the best split pos */
		for(int i=0; i<samples_num; i++)
		{
			/* the index in dataVec of current sample  */
			int sid = fppairs[i].pos;
			/* the index of node in current depth containing current sample */
			int cid = positions[fppairs[i].pos];
			/* the index of node's father node */
			int pid = q[cid].pid;
			
			if (tinfo.cnt0 >= min_children && q[pos].cnt - tinfo.cnt0 >= min_children && 
					sign(fea_list[i].first - tinfo.last_val) != 0) 
			{
				left_childs_sum[cid] += yVec[sid];
				left_childs_num[cid]++;

				/* computing the split gain of current split pos, split_gain = sum^2/N - sum1^2/L - sum2^2/R */
				float &ss0 = left_childs_sum[cid];
				float ss1 = treeNodes[pid].sum_y - ss0;
				
				float tempGain = treeNodes[pid].sum_sqr_y - (ss0*ss0 / left_childs_num[cid]) - 
					ss1*ss1 / (q[cid].cnt - left_childs_num[cid]);

				//printf("temp Gain is : %f \n", tempGain);
				
				SplitInfo &bsplit_info = splits_info[cid * sampled_fea_num +findex];
				
				if(tempGain < bsplit_info.err)
				{
						bsplit_info.err = tempGain;
						bsplit_info.bind = feature_id;
						bsplit_info.bsplit = fppairs[feature_id*samples_num + i].val;
						bsplit_info.cnt[0] = left_childs_num[cid];	
						bsplit_info.cnt[1] = q[cid].cnt - left_childs_num[cid];
						bsplit_info.sum_y[0] = ss0;
						bsplit_info.sum_y[1] = ss1;
						bsplit_info.sum_sqr_y[0] += square(yVec[sid]);
						bsplit_info.sum_sqr_y[1] = treeNodes[pid].sum_sqr_y - bsplit_info.sum_sqr_y[0];
				}
			}		
		}

	}


	/******************************************************************** 
	Description:   split all the nodes in the same depth after finding out best splitting 
	Input:           
		TNode *tree: current tree
		int &nodes_num: the number of nodes in current tree
		SplitInfo *best_split_info: the best splitting positions in current depth
		QNode *q: the nodes needing to splitting
		int *positions: the positions(index of tree node) of samples in current depth
		int &currLevel_childs_num: the number of nodes in current depth
		int samples_num:  the number of samples
	Output:
		new_q :  get the nodes needing to split in next depth 
		positions:  the positions(index of tree node) of samples 
	Return:         NULL
	**********************************************************************/ 
	void update_queue() {
		
		QNode *new_q = (QNode*) malloc(sizeof(QNode) *currLevel_childs_num * 2);
		//TNode new_node;
		int childs_num = 0;
		int *child_q_pos_first = (int*) malloc(sizeof (int) * currLevel_childs_num);
		int *child_q_pos_second =(int*) malloc(sizeof (int) * currLevel_childs_num);
		for (int i = 0; i < currLevel_childs_num; i++) {
			if (best_split_info[i].bind >= 0) {
				int ii = q[i].pid;
				tree[ii].ind = best_split_info [i].bind;                        
				tree[ii].splitval = best_split_info [i].bsplit;                
				tree[ii].ch[0] = nodes_num;
				tree[ii].ch[1] = nodes_num + 1;
				child_q_pos_first[i] =  childs_num;
				child_q_pos_second[i] =  childs_num + 1;

				for (int c = 0; c < 2; c++) {
					TNode &new_node = tree[nodes_num];
					new_node.ind = -1;
					new_node.value =  best_split_info[i].sum_y[c] / best_split_info[i].cnt[c];
					new_node.sum_y =  best_split_info[i].sum_y[c];
					new_node.sum_sqr_y =  best_split_info[i].sum_sqr_y[c];
					float err =  new_node.sum_sqr_y -  new_node.sum_y*new_node.sum_y/best_split_info[i].cnt[c];
					new_q[childs_num++] =  (QNode(nodes_num, best_split_info[i].cnt[c],  err));
					nodes_num++;
					//tree[nodes_num++] = (new_node);                    
				}                
			}
		}
		currLevel_childs_num = childs_num;

		for (int i = 0; i < samples_num; i++) {
			int &pos = positions[i];
			if (pos >= 0 && best_split_info[pos].bind >=  0) {
				if (dataVec[i*config.max_feature  + best_split_info[pos].bind] <=  best_split_info[pos].bsplit) {
					pos = child_q_pos_first[pos];
				} else {
					pos = child_q_pos_second[pos];
				}                
			} else pos = -1;
		}       
		free(q);
		free(child_q_pos_first);
		free(child_q_pos_second);
		q = new_q;
	}   

	/************************************************* 
	Description:   constructing regression tree 
	Input:           
		samples_ids: int[] : the samples index array used for constructing this tree
		samples_num: int  :the samples number 
	Output: 
		tree: TNode[] : the nodes array in this tree after constructing
		nodes_num: int 		: the nodes number in this tree after constructing
	Return:         NULL
	*************************************************/ 
	void buildDecisionTree(int *samples_ids, int samples_num, TNode *tree, int &nodes_num)
	{
		/* storing the data by column(features) in fppairs instead of by row(samples) */
		for (int i = 0; i < samples_num; i++)  
		{
			int sid = samples_ids[i];
			int fid = sid * config.max_feature;
			sqr_y_list[i] = square (yVec[sid]);
			sum_y += yVec[sid];
			sum_sqr_y += sqr_y_list[i];                
			positions[i] = 0;
			for(int j = 0; j < config.max_feature; j++)
			{
				fppairs[j * samples_num + i].pos = sid;
				fppairs[j * samples_num + i].val = dataVec[ fid + j ];
			}
		}

		/* sorting the features value by ascending */
		for (int i = 0; i < config.max_feature; i++)
			sort((fppairs + i*samples_num), (fppairs + (i+1) * samples_num), mySmaller);           
		
		/* adding the root node in this tree */
		nodes_num = 0;
		TNode &node = tree[nodes_num++];
		node.ind = -1;	
		node.value = sum_y / (samples_num ? samples_num : 1);
		node.sum_y = sum_y;
		node.sum_sqr_y = sum_sqr_y;

		/* add the first node needing to splitting */
		q[currLevel_childs_num++] = (QNode(0,  samples_num, sum_sqr_y- sum_y*sum_y/samples_num));  
		
		/* constructing the tree by breadth-first-search */
		for (int dep = 0; dep < config.depth;  dep++) {
			if (currLevel_childs_num == 0)  break; /* stop constructing if no nodes need to split */
			
			/* initial the split information for all nodes in the same depth */
			for (int i = 0; i <  currLevel_childs_num; i++) { 
				best_split_info[i].err =  MAX_ERROR;  
				best_split_info[i].bind =  -1;             
				for (int j = 0; j <  sampled_fea_num; j++) {                    
					splits_info[i*sampled_fea_num + j].err = MAX_ERROR;
					splits_info[i*sampled_fea_num + j].bind = -1;
					splits_info[i*sampled_fea_num + j].sum_sqr_y[0] = 0;
				}
			}

			/* find the best splitting positions given the feature(feas) array for current depth */
			for (int findex = 0; findex <sampled_fea_num; findex++)
			{
				/* randomly select feature to find split position */
				int feature_id = (int) rand() % config.max_feature;
				find_splits( &(fppairs[feature_id * samples_num]), samples_num, feature_id, positions, 
						sqr_y_list, q, splits_info, tree, sampled_fea_num, currLevel_childs_num, findex);
			}

			/* merge split positions on all features to a best splitting */
			for (int i = 0; i < currLevel_childs_num; i++) {
				SplitInfo &bspinfo =  best_split_info[i];
				for (int j = 0; j <  sampled_fea_num; j++)
				{
					SplitInfo &tempSplitInfo = splits_info[i*sampled_fea_num + j];
					if (tempSplitInfo.bind >= 0 && (bspinfo.bind < 0 || bspinfo.err >  tempSplitInfo.err))
						bspinfo.update(tempSplitInfo);
				}
			}
			/* split nodes to further depth */
			update_queue(tree, nodes_num, best_split_info, q, positions, 
					currLevel_childs_num, samples_num);

		}
	}
}

/* Random  Forest */
class Forest
{
public:
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
	void buildForest(float *dataVec, float *yVec)
	{        
		srand((unsigned)time(NULL));
		int *samples_ids = (int *)malloc(sizeof(int) * config.data_num);
		for (int i = 0; i < config.tree_num; i++)
		{
			/* randomly sampling the data to construct tree */
			for(int j = 0; j < config.data_num; j++)
				samples_ids[j] = (int)rand() % config.data_num;
			buildDecisionTree(samples_ids, config.data_num, &(forestNodes[i*MAX_NODES_NUM]), treeNodesNum[i]);
		}
	}
}

/********************************************************************
test function, get the test samples
*********************************************************************/
void testData(ForestConfig &config)
{
	config.data_num = 100000;	 /* samples number. */
	config.max_feature = 64; /* sample's features number */
	config.tree_num = 100;	 /* trees number in forest */
	config.depth = 12;		 /* maximum depth in trees */
	config.min_children = 4;/* minimum samples number in leaf nodes */
	config.bootstrap = -0.1; /* sampling ratio for samples */
	config.nthread = 1;	 /* maximum parallel threads number */

	dataVec = (float*) malloc(sizeof(float ) * config.data_num * config.max_feature);
	yVec = (float *) malloc(sizeof( float) * config.data_num);

	/* fill the test data */
	for(int i=0; i<config.data_num; i++)
	{
		yVec[i] = i + 1.0f;
		for(int j=0; j<config.max_feature; j++)
		{
			dataVec[i*config.max_feature + j] = i + 1.0f;
		}
	}

}

int main(int argc, char* argv[])
{
	//read_conf_file(config, argc, argv);
    /* config is the configuration information */
	testData(config);
	
	double start_time, end_time;
	
	buildForest(dataVec, yVec);

	free(dataVec);
	free(yVec);

    printf("forest building feature parallel executing time: %f\n", end_time - start_time);
	
}
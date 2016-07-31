# mic-RF
I want to implement random forest algorithm on Intel Xeon Phi(MIC).

算法流程：

对样本集进行采样  ————>  构造决策树 （n棵）————>  形成森林

构造决策树流程： 宽度优先。  

重要的变量说明：


int *samples_ids： 记录采样得到的样本集
float *sqr_y_list:

splits_info: 如何存放的？ 先存储一个节点的所有属性
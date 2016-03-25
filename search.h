#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "train.h"

using namespace cv;
using namespace std;


struct node
{
	float data;
	int no;
};


class Searcher
{
public:
	Searcher();
	~Searcher();
	void createSearcher();

	void readDict(FileStorage &fs1);  //
	node*  findMinDistance(char *testImgPath, int numOfPictures,FileStorage  &fs2);

private:
	int numOfPictures;
	float scale;
	FileStorage fs1;
	FileStorage fs2;
				   
	Ptr<FeatureDetector> features;
	Ptr<DescriptorExtractor> descriptors;
	Ptr<DescriptorMatcher> matcher;

	BOWKMeansTrainer  *trainer;  
	BOWImgDescriptorExtractor *bowDE;

	Trainer imgTraier;
};

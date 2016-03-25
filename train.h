#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

class Trainer
{
public :
	Trainer();
	~Trainer();
	
	void createTrainer(int dictionarySize);

	void trainVocabulary(int numOfPictures);
	void trainBow(int numOfPictures);
	
																	 
private:
	int numOfPictures;
	char filename[100];
	float scale;
	int dictionarySize;
	//vector<KeyPoint> keypoints[numOfPictures];
	vector<KeyPoint> keypoints;

	Ptr<FeatureDetector> features;
	Ptr<DescriptorExtractor> descriptors;
	Ptr<DescriptorMatcher> matcher;

	BOWKMeansTrainer  *trainer;  
	BOWImgDescriptorExtractor *bowDE; 
};
#include "train.h"
#include "search.h"


int cmp(const void *a,const void *b)
{
	return (*(node *)a).data > (*(node *)b).data ? 1 : -1; 
}

Searcher::Searcher()
{
	
}

Searcher::~Searcher()
{
}

void Searcher::createSearcher()
{
	scale = 2;
	initModule_nonfree();//使用SIFT/SURF create之前，必须先initModule_<modulename>();
	features = FeatureDetector::create("SURF");
	descriptors = DescriptorExtractor::create("SURF");
	matcher = DescriptorMatcher::create("FlannBased");	  
	bowDE = new BOWImgDescriptorExtractor(descriptors,matcher);
	
	imgTraier.createTrainer(2000);
}


void Searcher::readDict(FileStorage &fs1)
{
	/*
	FileStorage fs1("dictionary.yml", FileStorage::READ);	 	
	FileStorage fs2("bow_descriptor.yml", FileStorage::READ);
	if(!fs1.isOpened() || !fs2.isOpened())
	{
		imgTraier.trainVocabulary(numOfPictures);
		imgTraier.trainBow(numOfPictures);
	}
	*/

	Mat centers;

	fs1["vocabulary"]>>centers;
	fs1.release();	
	bowDE->setVocabulary(centers);//设置一个视觉词典
}

node* Searcher::findMinDistance(char *testImgPath, int numOfPictures,FileStorage &fs2)
{
	vector<KeyPoint> testimg_key;
	Mat testbowDescriptor;

	Mat testImg = imread(testImgPath,1);
	if (!testImg.data)
	{
		cout<<"shit,fail to read the testimage!"<<endl;
	}	
	resize(testImg,testImg,Size(testImg.cols/scale,testImg.rows/scale));

	cout<<">Extracting keypoints of the test image..."<<endl;
	features->detect(testImg,testimg_key);
	bowDE->compute(testImg,testimg_key,testbowDescriptor);
	
	normalize(testbowDescriptor,testbowDescriptor,1,0,CV_L2);
	cout<<"testbowDescriptor" <<testbowDescriptor.size()<<"    "<<testbowDescriptor.type()<<endl;
	char tagname[100]; 
	vector<double>distance;
	//double distance[70];
	Mat  bow_descriptor;
	struct node *nodeNum;
	nodeNum = (node*)malloc(sizeof(node)*numOfPictures);

#pragma omp parallel for
	for (int i=0;i<numOfPictures;i++)
	{
		sprintf(tagname,"BOW%d",i);
		fs2[tagname]>>bow_descriptor;	 //读出每一张库图的字典表示	 
		//normalize(bow_descriptor,bow_descriptor,1,0,CV_L2);
		//cout<<"bow_descriptor" <<bow_descriptor.size()<<"    "<<bow_descriptor.type()<<endl;	
		//distance.push_back(norm(testbowDescriptor,bow_descriptor,CV_L2));
		//cout<<"distance"<<i<<" = "<<distance[i]<<endl;
		nodeNum[i].data = norm(testbowDescriptor,bow_descriptor,CV_L2);
		nodeNum[i].no = i;
	}
	qsort(nodeNum,numOfPictures,sizeof(nodeNum[0]),cmp); //快速排序函数
	return 	 nodeNum;
}
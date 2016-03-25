#include <iostream>	
#include <omp.h>
#include <io.h>
#include <windows.h>
#include <shlwapi.h>
#include "train.h"
#include "search.h"
#include "opencv2\opencv.hpp"
#include "opencv2\nonfree\nonfree.hpp"

const int numOfPictures = 70;
const int dictionarySize = 2000;

void searchMatch(Searcher &imgSearcher,
	FileStorage &fs,
	_finddata_t file,
	int &matchNum_1,int &matchNum_5,int &matchNum_10,
	int top1,int top5,int top10);

int main()
{
	int frameNum = 1;
	int num = 0;
	int top1 = 1;
	int top5 = 5;
	int top10 = 10;
	int matchNum_1 = 0;
	int matchNum_5 = 0;
	int matchNum_10 = 0;

	Trainer imgTraier;
	imgTraier.createTrainer(dictionarySize);
	
	Searcher imgSearcher;
	imgSearcher.createSearcher();

	FILE *f1,*f2;
	f1 = fopen("dictionary.yml","r");
	f2 = fopen("bow_descriptor.yml","r");
	if (NULL == (f1 && f2)) 
	{
		cout<<"没有训练数据！正在进行训练..."<<endl;
		imgTraier.trainVocabulary(numOfPictures);
		imgTraier.trainBow(numOfPictures);
	}
	fclose(f1);
	fclose(f2);

	FileStorage fs1("dictionary.yml", FileStorage::READ);	 	
	FileStorage fs2("bow_descriptor.yml", FileStorage::READ);
	
	if(!fs1.isOpened() || !fs2.isOpened())
	{
		cout<<"打开训练数据失败！"<<endl;
		return 0;
	}
	
	imgSearcher.readDict(fs1);//读取存储的词汇，设置成字典集	  

	double t= (double)getTickCount();

	char* search_path ="testimg\\*.jpg";
	_finddata_t file;  
	long lf;  
	//输入文件夹路径  
	if((lf = _findfirst(search_path, &file))==-1)  
		cout<<"Not Found!"<<endl;  
	else{  	
		searchMatch(imgSearcher,fs2,file,matchNum_1,matchNum_5,matchNum_10,top1,top5,top10);
		while(_findnext( lf, &file)==0){  
			searchMatch(imgSearcher,fs2,file,matchNum_1,matchNum_5,matchNum_10,top1,top5,top10);
			cout<<"frameNum:"<<++frameNum<<endl;
			cout<<"matchNum_1:"<<matchNum_1<<endl;
			cout<<"matchNum_5:"<<matchNum_5<<endl;
			cout<<"matchNum_10:"<<matchNum_10<<endl<<endl;
		}  
	}  
	_findclose(lf); 

	t = (double)getTickCount() - t;
	cout<<"CostTime:"<<t*1000/getTickFrequency()<<"ms"<<endl;

	cout<<"end"<<endl;
	waitKey();
	return 0;
}

void searchMatch(Searcher &imgSearcher,
	FileStorage  &fs,
	_finddata_t file,
	int &matchNum_1,int &matchNum_5,int &matchNum_10,
	int top1,int top5,int top10)
{
	char testimgname[100];
	string imgname;
	int num = 0;
	string numStr = file.name;
	numStr = numStr.substr(0,strlen(file.name)-4);
	stringstream ss;
	ss << numStr;
	ss >> num;  //此时的num就是待测试图片和库图片的标号
	//sprintf(testimg, "testing_half/photo2/标号/%s.jpg", numStr.data());
	cout<<endl<<"the testimage name : "<<numStr.data()<<".jpg "<<endl;
	sprintf(testimgname, "testimg/%s.jpg", numStr.data());

	struct node *nodeNum;
	nodeNum = imgSearcher.findMinDistance(testimgname, numOfPictures,fs);
	cout<<"num = "<<num<<endl;
	if(num == nodeNum[0].no)
	{
		//cout<<"top1 imgnum "<<nodeNum[0].no<<endl;
		matchNum_1++;
		//Mat img = imread(testimg,1);
		//imwrite("testimg/top1/"+numStr+".jpg",img);
	}
	for(int i = 0;i < top5;i++)
	{
		//cout<<"top5 imgnum "<<nodeNum[i].no<<endl;
		//histogram_top5[nodeNum[i].no-1]++;
		if(num == nodeNum[i].no) 
		{
			matchNum_5++;
			//Mat img = imread(testimg,1);
			//imwrite("testing_half/photo2/标号/top5/"+numStr+".jpg",img);
			break;
		}
	}
	for(int i = 0;i < top10;i++)
	{
		cout<<"top10 imgnum "<<nodeNum[i].no<<endl;
		//histogram_top10[nodeNum[i].no-1]++;
		if(num == nodeNum[i].no) 
		{
			matchNum_10++;
			//Mat img = imread(testimg,1);
			//imwrite("testing_half/photo2/标号/top10/"+numStr+".jpg",img);
			break;
		}
	}
	delete []nodeNum;
}

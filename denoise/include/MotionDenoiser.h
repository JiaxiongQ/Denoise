#include "MeshFlow.h"
#include "VideoIO.h"
#include "gridTracker.h"
#include "time.h"
#include "Fast_klt.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <queue>
#include <deque>  
#include <fstream>
#include <eigen3/Eigen/Dense>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/core/ocl.hpp>
#include "opencv2/core/internal.hpp"
//#define HAVE_TBB
#include <sys/time.h>
#define N  50 //210:50//311:20
#define tau 20//210:50(static)/20(dynamic)//311:80
#define PI 3.1415926535898
#define sqrtPI std::sqrt(PI / 2)
#define ps 3
#ifndef __MotionDenoiser__
#define __MotionDenoiser__

class MotionDenoiser{
	MeshFlow meshflow;
	gridTracker gt;
public:
	int preKFtrackedP,preKFindex;
	int m_height;
	int m_width;
	int m_frameNum;
	int offsetL = 0,offsetR = N + 1;
	cv::Size m_size;
	double m_fps;
	bool flag = true;
	int c;
	float tempMean1,tempMean2;
	double times = 0.0,times1 = 0.0,times2 = 0.0;
	double time1;
	bool startFlag = false;
	bool endFlag = false;
	vector<cv::Point2f> sourceFeatures;
	vector<cv::Point2f> targetFeatures;
	vector<unsigned char> KFindexs;
	vector<int> KfStartNum;
	vector<int> KfEndNum;
	cv::UMat allOnes;
public:
	MotionDenoiser(char* name);
	void Execute();
	void SaveResult(char* name);
	double get_wall_time();
	void showTemp(int m);
private:
	vector<cv::Mat> m_frames,m_framesG;
	vector<cv::Mat> map_X,map_Y;
	vector<cv::UMat> map_XU,map_YU;
	vector<cv::UMat> Uframes,Udst;
	vector<cv::UMat> map_XUs,map_YUs;
	cv::UMat map_XUt,map_YUt;
	cv::UMat m_mask;
	cv::UMat mask_temp;
	cv::UMat m_dst_tempUt;
	cv::UMat m_Counter_adderUt;
	cv::UMat m_temp; 
	cv::UMat m_mapedX, m_mapedY;
	cv::Mat formatX, formatY;
	cv::UMat formatXU, formatYU;
	cv::UMat temp;
	cv::UMat temp1;
	cv::UMat temp2;
	cv::UMat temp3;
	cv::UMat temp4;
	cv::UMat temp5;
	cv::UMat temp6;
	cv::UMat temp7;
	cv::UMat temp8;
	cv::UMat tempR;
	vector<cv::UMat> tempRs;
private:
        void KfDetection(int reference);
	void MotionEstimation();
	void Judge(int reference);
	void TargetFrameBuildNew(int reference);
	void FusionNew(int m,int k,bool kfFlag,int start);
};
#endif
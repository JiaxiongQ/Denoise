#include "VideoIO.h"

vector<cv::Mat> GetFrames(char* name,double &fps){
	//printf("Read Video\n");
	cv::VideoCapture capture(name);
	// check if video successfully opened
	if (!capture.isOpened())
	{
		cerr << "The video can not open!";
		exit(0);
	}
	fps =  capture.get(CV_CAP_PROP_FPS);

	cv::Mat frame, frame_copy;
	vector<cv::Mat> dst;
	int frame_count = 0;
	cv::Mat m_framesGray;
	int counter = 0;
	while (capture.read(frame))
	{
		frame_copy = frame.clone();
	        dst.push_back(frame_copy);
		counter++;
		if(counter == 150)
			break;
	}
	capture.release();
	return dst;
}

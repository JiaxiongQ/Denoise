#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "MeshFlow.h"
// #include "MotionDenoiser.h"
#include "DirectWarpDenoiser.h"
#include <time.h>
#include <omp.h>

int main(){
   /*
    if (!cv::ocl::haveOpenCL())
    {
        cout << "OpenCL is not avaiable..." << endl;
    }
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        cout << "Failed creating the context..." << endl;
    }

    // In OpenCV 3.0.0 beta, only a single device is detected.
    cout << context.ndevices() << " GPU devices are detected." << endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        cout << "name                 : " << device.name() << endl;
        cout << "available            : " << device.available() << endl;
        cout << "imageSupport         : " << device.imageSupport() << endl;
        cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
        cout << endl;
    }

    // Select the first device
    cv::ocl::Device(context.device(0));
    */
   
 
   vector<char*> names;
   vector<char*> outNames;
   for (int i = 2; i <= 2; i++){
	   char* name = new char[1024];
	   char* outname = new char[1024];

	   sprintf(name, "/home/xxd/Desktop/qjxtemp/MeshFlow_Video_Denoising-masterOff/build/Qiu/Q611m.avi",i);
	   sprintf(outname, "/home/xxd/Desktop/qjxtemp/MeshFlow_Video_Denoising-masterOff/build/Qiu/Q611wowsout.avi",i);

	   names.push_back(name);   
	   outNames.push_back(outname);
	}

	for (int i = 0; i < names.size(); i++){
// 	   MotionDenoiser denoiser(names[i]);
	   DirectWarpDenoiser denoiser(names[i]);
// 	   denoiser.Execute();
	   denoiser.SaveResult(outNames[i]);
	}
	/*
	cv::Mat src1 = cv::imread("out.png");
	cv::Mat src1Out = src1(Rect(20, 20, src1.cols - 60, src1.rows -60));
	//cv::Mat src1Out = src1(Rect(0, 0, src1.cols - 20, src1.rows -20));
	cv::imwrite("out.png",src1Out);
	*/
	return 0;
}
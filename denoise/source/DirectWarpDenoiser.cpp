#include "DirectWarpDenoiser.h"
double DirectWarpDenoiser::get_wall_time(){
	struct timeval time;
	if(gettimeofday(&time,NULL)){
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}



DirectWarpDenoiser::DirectWarpDenoiser(char* name){
	cv::ocl::setUseOpenCL(true);
	
	m_frames = GetFrames(name,m_fps);
	
	m_size = m_frames[0].size();
	m_height = m_size.height;
	m_width = m_size.width;
	m_frameNum = m_frames.size();
	
	Uframes.resize(m_frameNum);
	m_framesG.resize(m_frameNum);
	for (int i = 0; i < m_frameNum; i++){
		Uframes[i] = m_frames[i].getUMat(ACCESS_READ);
		cv::cvtColor(m_frames[i],m_framesG[i],cv::COLOR_RGB2GRAY);
	}

	Udst.resize(m_frameNum);
	for (int i = 0; i < Udst.size(); i++) Udst[i].create(m_size, CV_8UC3);
	
	map_XUs.resize(m_frameNum);
	map_YUs.resize(m_frameNum);
	map_X.resize(m_frameNum);
	map_XU.resize(m_frameNum);
	tempRs.resize(m_frameNum);
	for (int i = 0; i < map_X.size(); i++){
		map_X[i].create(m_size, CV_32FC1);
		map_XU[i].create(m_size, CV_32FC1);
		map_XUs[i].create(m_size, CV_32FC1);
		tempRs[i].create(m_size,CV_8UC3);
	}
	map_Y.resize(m_frameNum);
	map_YU.resize(m_frameNum);
	for (int i = 0; i < map_Y.size(); i++){
		map_Y[i].create(m_size, CV_32FC1);
		map_YU[i].create(m_size, CV_32FC1);//times += time1;
		map_YUs[i].create(m_size, CV_32FC1);
	}
	temp_map_XUs.resize(2 * N);
	for (int i = 0; i < 2*N; i++){
		temp_map_XUs[i].create(m_size, CV_32F);
		temp_map_XUs[i].setTo(1.0);
	}

	temp_map_YUs.resize(2 * N);
	for (int i = 0; i < 2*N; i++){
		temp_map_YUs[i].create(m_size, CV_32F);
		temp_map_YUs[i].setTo(1.0);
	}
	map_XUt.create(m_size, CV_32FC1);
	map_XUt.setTo(0.0);
	map_YUt.create(m_size, CV_32FC1);
	map_YUt.setTo(0.0);
	
	Uframes[0].convertTo(m_dst_tempUt,CV_32FC3);
	m_Counter_adderUt.create(m_size, CV_32FC3);
	m_Counter_adderUt.setTo(1.0);
	allOnes.create(m_size, CV_32FC3);
	allOnes.setTo(1.0);
	
	m_mapedX.create(m_size, CV_32FC1);
	m_mapedY.create(m_size, CV_32FC1);
	
	m_temp.create(m_size, CV_32FC3);
	m_mask.create(m_size, CV_32FC3);
	mask_temp.create(m_size,CV_8UC3);
	
	formatX = cv::Mat::zeros(m_size, CV_32FC1);
	for (int i = 0; i < formatX.rows; i++)
		for (int j = 0; j < formatX.cols; j++)
			formatX.at<float>(i, j) = j;
	formatY = cv::Mat::zeros(m_size, CV_32FC1);
	for (int i = 0; i < formatY.rows; i++)
		for (int j = 0; j < formatY.cols; j++)
			formatY.at<float>(i, j) = i;
	formatXU = formatX.getUMat(ACCESS_READ);
	formatYU = formatY.getUMat(ACCESS_READ);
	
	temp.create(m_size,CV_8UC3);
	temp1.create(m_size,CV_8UC3);
	tempR.create(m_size,CV_8UC3);
	temp2.create(m_size,CV_8UC3);
	temp3.create(m_size,CV_32FC3);
	temp4.create(m_size,CV_32FC3);
	temp5.create(m_size,CV_32FC1);
	temp6.create(m_size,CV_32FC1);
	temp7.create(m_size,CV_32FC1);
	temp8.create(m_size,CV_8UC1);
	
	KFindexs.resize(m_frameNum);
	KfStartNum.resize(m_frameNum);
	KfEndNum.resize(m_frameNum);
	for (int i = 0; i < m_frameNum; i++){
		KFindexs[i] = 0;
		KfStartNum[i] = 0;
		KfEndNum[i] = m_frameNum - 1;
	}
	tempR = Uframes[0];
	
	meshflow.m_height = m_height;
	meshflow.m_width = m_width;
	meshflow.Initialize();
	gt.trackerInit(m_framesG[0]);
}

void DirectWarpDenoiser::MotionEstimation(){
	for (int i = 1; i < m_frameNum; i++){
	    gt.Update(m_framesG[i-1], m_framesG[i]);
	    meshflow.ReInitialize();
// 	    meshflow.SetFeature(gt.trackedFeas,gt.preFeas);
	    meshflow.SetFeature(gt.preFeas, gt.trackedFeas);
	    meshflow.Execute();
	    meshflow.GetMotions(map_X[i-1], map_Y[i-1]);
	    
	    map_XU[i-1] = map_X[i-1].getUMat(ACCESS_READ);
	    map_YU[i-1] = map_Y[i-1].getUMat(ACCESS_READ);
    
	    map_XU[i-1].copyTo(map_XUs[i-1]);
	    map_YU[i-1].copyTo(map_YUs[i-1]);
	}
}


void DirectWarpDenoiser::Execute(){
	
	int reference = 5;
	
	double startTime = get_wall_time();
	MotionEstimation();
	for (int i = 1; i < m_frameNum; i++){
	  
	  AbsoluteMotion(i);
	
	  TargetFrameBuild(i);
	  
	}
	double endTime = get_wall_time();
	cout << "duration(s/frame): " << (endTime - startTime  - times) / (m_frameNum) << endl;
}

void DirectWarpDenoiser::AbsoluteMotion(int reference){
      for (int i = reference - N, k = 0; i < reference && k < N; i++,k++){
		if (i >= 0){
			temp_map_XUs[k] = map_XUs[i];
			temp_map_YUs[k] = map_YUs[i];
			for (int j = i + 1 ; j < reference; j++){
				cv::add(temp_map_XUs[k],map_XUs[j],temp_map_XUs[k]);
				cv::add(temp_map_YUs[k],map_YUs[j],temp_map_YUs[k]);
			}
		}
	}
      for (int i = reference + N - 1, k = 2 * N - 1; i >= reference&&k >= N; i--, k--){
		if (i < m_frames.size() - 1){

			cv::multiply(map_XUs[i],-1.0,temp_map_XUs[k]);
	
			cv::multiply(map_YUs[i],-1.0,temp_map_YUs[k]);
			for (int j = i - 1; j >= reference; j--){
				cv::subtract(temp_map_XUs[k],map_XUs[j],temp_map_XUs[k]);
				cv::subtract(temp_map_YUs[k],map_YUs[j],temp_map_YUs[k]);
			}
		}
	}
}

void DirectWarpDenoiser::TargetFrameBuild(int reference){
	Uframes[reference].convertTo(m_dst_tempUt,CV_32FC3);
	m_Counter_adderUt.setTo(1.0);
	tempR = Uframes[reference];
	
	//left part
	for (int k = reference - N, m = 0; k < reference&&m < N; k++, m++){
		if (k >= 0){
			cv::add(temp_map_XUs[m],formatXU,m_mapedX);
			cv::add(temp_map_YUs[m],formatYU,m_mapedY);
			cv::remap(Uframes[k], temp, m_mapedX, m_mapedY, cv::INTER_LINEAR);

			cv::absdiff(tempR, temp, temp2);
			cv::compare(temp2,tau, mask_temp, cv::CMP_LE);
			mask_temp.convertTo(m_mask,CV_32FC3,1.0/255);
			cv::add(m_Counter_adderUt,m_mask,m_Counter_adderUt);
			
			temp.convertTo(m_temp,CV_32FC3);
			cv::multiply(m_temp,m_mask,temp3);
			cv::add(m_dst_tempUt,temp3,m_dst_tempUt);
			
		}
	}
	//right part
	for (int k = reference + 1, m = N; k < reference + N + 1 && m < 2 * N; k++, m++){
		if (k < m_frames.size()){
			cv::add(temp_map_XUs[m],formatXU,m_mapedX);
			cv::add(temp_map_YUs[m],formatYU,m_mapedY);
			cv::remap(Uframes[k], temp, m_mapedX, m_mapedY, cv::INTER_LINEAR);

			cv::absdiff(tempR, temp, temp2);
			cv::compare(temp2,tau, mask_temp, cv::CMP_LE);
			mask_temp.convertTo(m_mask,CV_32FC3,1.0/255);
			cv::add(m_Counter_adderUt,m_mask,m_Counter_adderUt);
			
			temp.convertTo(m_temp,CV_32FC3);
			cv::multiply(m_temp,m_mask,temp3);
			cv::add(m_dst_tempUt,temp3,m_dst_tempUt);
			
		}
	}
	
	cv::divide(m_dst_tempUt,m_Counter_adderUt,temp4);
	temp4.convertTo(Udst[reference],CV_8UC3);
}


void DirectWarpDenoiser::SaveResult(char* name){
	cv::VideoWriter outVideoWriter;
	cout << "fps: " << m_fps << endl;
	cout << "fNum: " << m_frameNum << endl; 
	outVideoWriter.open(name, CV_FOURCC('X', 'V', 'I', 'D'),m_fps,m_size);
	for (int i = 0; i < m_frameNum; i++){
// 		Uframes[i].convertTo(Udst[reference],CV_8UC3);
		if(i == 40)
			cv::imwrite("/out0.png",Udst[i].getMat(ACCESS_READ));
		outVideoWriter << Udst[i].getMat(ACCESS_READ);
	}
	outVideoWriter.~VideoWriter();
}
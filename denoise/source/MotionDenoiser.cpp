#include "MotionDenoiser.h"

double  MotionDenoiser::get_wall_time(){
	struct timeval time;
	if(gettimeofday(&time,NULL)){
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

MotionDenoiser::MotionDenoiser(char* name){
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
	
	map_XUs.resize(m_frameNum - 1);
	map_YUs.resize(m_frameNum - 1);
	map_X.resize(m_frameNum - 1);
	map_XU.resize(m_frameNum - 1);
	tempRs.resize(m_frameNum - 1);
	for (int i = 0; i < map_X.size(); i++){
		map_X[i].create(m_size, CV_32FC1);
		map_XU[i].create(m_size, CV_32FC1);
		map_XUs[i].create(m_size, CV_32FC1);
		tempRs[i].create(m_size,CV_8UC3);
	}
	map_Y.resize(m_frameNum - 1);
	map_YU.resize(m_frameNum - 1);
	for (int i = 0; i < map_Y.size(); i++){
		map_Y[i].create(m_size, CV_32FC1);
		map_YU[i].create(m_size, CV_32FC1);//times += time1;
		map_YUs[i].create(m_size, CV_32FC1);
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

void MotionDenoiser::KfDetection(int reference){//tennis:89,148
	/*
	if(reference > (preKFindex + N)){
		int curKFtrackedP = gt.trackedFeas.size();
		//float ttt = curKFtrackedP * 1.0/ preKFtrackedP;
		//cout << ttt << endl;
		if(curKFtrackedP > 500 && (curKFtrackedP * 1.0 / preKFtrackedP < 0.9)){
			KFindexs[reference] = 1;
		}
	}
	*/
	/*
	for(int i = 0;i <  89;i+= 5){
		KFindexs[i] = 1;
	}

	for(int i = 90;i <  145;i+=5){
		KFindexs[i] = 1;
	}
	
	//for(int i = 61;i < 72;i++){
	//	KFindexs[i] = 2;
	//}
	KFindexs[89] = 2;
	KFindexs[148] = 2;
	KFindexs[149] = 2;
	KFindexs[100] = 1;
	*/
	/*
	for(int i = 0;i <  150;i+=3){
		KFindexs[i] = 1;
	}
	KFindexs[125] = 2;
	KFindexs[124] = 2;
	KFindexs[119] = 2;
	KFindexs[118] = 2;
	for(int i = 116;i >=  112;i--)
		KFindexs[i] = 2;
	KFindexs[107] = 2;
	KFindexs[74] = 2;
	KFindexs[73] = 2;
	KFindexs[65] = 2;
	KFindexs[62] = 2;
	KFindexs[61] = 2;
	KFindexs[50] = 2;
	*/
	for(int i = 0;i <=  m_frameNum;i+=5)
		KFindexs[i] = 1;
}

void MotionDenoiser::MotionEstimation(){
	int KftempS = 0;
	const int endNum = m_frameNum - 1;
	int KftempE = endNum;
	//KfDetection(0);
	preKFtrackedP = gt.allFeas.size();
	preKFindex = 0;
	for (int i = 1; i < m_frameNum; i++){
		gt.Update(m_framesG[i-1], m_framesG[i]);
		
		if(i > N){
			if(i == KftempS){
				KftempE = endNum;
				preKFindex = i;
				//gt.trackerInit(m_framesG[i]);
				//preKFtrackedP = gt.allFeas.size();
				preKFtrackedP = gt.preFeas.size();
			}
			else if(i == KftempS + 1){
				map_XUt.setTo(0.0);
				map_YUt.setTo(0.0);
			}
		}
		
		if(i < m_frameNum - N &&  KFindexs[i + N] == 1)
		{
			KftempS = i + N;
			KftempE = i + N;
			
		}
		
		KfStartNum[i] = KftempS;
		KfEndNum[i] = KftempE;
		//gt.trackerInit(m_framesG[0]);
		//gt.Update(m_framesG[0], m_framesG[i]);
		
		//cout<< i << " " << gt.trackedFeas.size() << endl;
		meshflow.ReInitialize();
		meshflow.SetFeature(gt.preFeas, gt.trackedFeas);
		meshflow.Execute();
		meshflow.GetMotions(map_X[i-1], map_Y[i-1]);
		
// 		double t1 = get_wall_time();
		map_XU[i-1] = map_X[i-1].getUMat(ACCESS_READ);
		map_YU[i-1] = map_Y[i-1].getUMat(ACCESS_READ);
// 		double t2 = get_wall_time();
// 		times += (t2 - t1);
		 
		cv::subtract(map_XUt,map_XU[i - 1],map_XUt);
		cv::subtract(map_YUt,map_YU[i - 1],map_YUt);
		//cv::multiply(-1.0,map_XU[i - 1],map_XUt);
		//cv::multiply(-1.0,map_YU[i - 1],map_YUt);
		
		map_XUt.copyTo(map_XUs[i-1]);
		map_YUt.copyTo(map_YUs[i-1]);
	}
}

void MotionDenoiser::Judge(int reference){
	startFlag = reference == KfStartNum[reference]?true:false;
	endFlag =  reference > (KfEndNum[reference] - N)?true:false;
}

void MotionDenoiser::Execute(){
	
	double startTime = get_wall_time();
	MotionEstimation();
	
	for (int i = 0; i < m_frameNum; i++){
		//cout << i << endl;
		if(KFindexs[i] == 2){
			Uframes[i].copyTo(Udst[i]);
			continue;
		}
		Judge(i);
		if(KFindexs[i] == 1){
			Uframes[i].convertTo(m_dst_tempUt,CV_32FC3);
			m_Counter_adderUt.setTo(1.0);
			tempR = Uframes[i];
		}
		TargetFrameBuildNew(i);
	}
	double endTime = get_wall_time();
	cout << "duration(s/frame): " << (endTime - startTime  - times) / (m_frameNum) << endl;
}

inline void MotionDenoiser::FusionNew(int m,int k,bool kfFlag,int start){
	if(!kfFlag){
		cv::add(map_XUs[m],formatXU,m_mapedX);
		cv::add(map_YUs[m],formatYU,m_mapedY);
	}
	else{
		cv::subtract(map_XUs[m],map_XUs[start - 1],map_XUt);
		cv::subtract(map_YUs[m],map_YUs[start - 1],map_YUt);
		cv::add(map_XUt,formatXU,m_mapedX);
		cv::add(map_YUt,formatYU,m_mapedY);
	}
	cv::remap(Uframes[k], temp, m_mapedX, m_mapedY, cv::INTER_LINEAR);
	
	if(!kfFlag) temp.copyTo(tempRs[m]);
	
	cv::absdiff(tempR, temp, temp2);
// 	cv::blur(temp2, temp2, Size(ps, ps));
	cv::compare(temp2,tau, mask_temp, cv::CMP_LE);
	mask_temp.convertTo(m_mask,CV_32FC3,1.0/255);
	cv::add(m_Counter_adderUt,m_mask,m_Counter_adderUt);
	
	//cv::add(m_Counter_adderUt,allOnes,m_Counter_adderUt);
	
	temp.convertTo(m_temp,CV_32FC3);
	cv::multiply(m_temp,m_mask,temp3);
	cv::add(m_dst_tempUt,temp3,m_dst_tempUt);
	
	//temp.convertTo(m_temp,CV_32FC3);
	//cv::add(m_dst_tempUt,m_temp,m_dst_tempUt);
}

void MotionDenoiser::TargetFrameBuildNew(int reference){
	if(startFlag){
		int temp = 2;
		if(reference == 0){
			temp = 0;
		}
		else if(reference == (m_frameNum - 1)){
			temp = 1;
		}
		switch(temp){
			case 0:
				offsetL = 0;
				offsetR = N + 1;
				break;
			case 1:
				offsetL = N + 1;
				offsetR = 0;
				break;
			default: 
				offsetL = N + 1;
				offsetR = N + 1;
				break;
		}
		    
		int start = KfStartNum[reference];
		for (int k = 1 + start - offsetL,m = start - offsetL; k <  start + offsetR; k++,m++){
			if(k>0&&k<m_frameNum){
				if(k < start)
					FusionNew(m,k,true,start);
				else if(k != start)
					FusionNew(m,k,false,0);
			}
		} 
		cv::divide(m_dst_tempUt,m_Counter_adderUt,temp4);
		temp4.convertTo(Udst[reference],CV_8UC3);
	}
	else{
		if(endFlag){
			temp = temp1; 
		}
		else{
			tempR = tempRs[reference - 1];
			int k = reference + N;
			int m = k - 1;
			FusionNew(m,k,false,0);
			cv::divide(m_dst_tempUt,m_Counter_adderUt,temp4);
			temp4.convertTo(temp,CV_8UC3);
			temp1 = temp;
		}
		
		cv::multiply(-1.0,map_XUs[reference - 1],temp5);
		cv::add(temp5,formatXU,m_mapedX);
		cv::multiply(-1.0,map_YUs[reference - 1],temp6);
		cv::add(temp6,formatYU,m_mapedY);
		
		//deal with border...
// 		cv::compare(m_mapedX, 0.0, temp5, cv::CMP_LT);
// 		cv::compare(m_mapedX, 1.0 * m_width - 5.0, temp6, cv::CMP_GT);
// 		cv::add(temp5,temp6,temp7);
// 		cv::compare(m_mapedY, 0.0, temp5, cv::CMP_LT);//
// 		cv::add(temp7,temp5,temp6);
// 		cv::compare(m_mapedY, 1.0 * m_height - 5.0, temp5, cv::CMP_GT);
// 		cv::add(temp6,temp5,temp7);
		
		cv::remap(temp, Udst[reference], m_mapedX, m_mapedY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(1,1,1));

		//deal with border...
		//cv::compare(Udst[reference], cv::Scalar(1,1,1), temp2, cv::CMP_EQ);
		//temp7.convertTo(temp2,CV_8UC1);
// 		Uframes[reference].copyTo(Udst[reference],temp7);
	}
}

void MotionDenoiser::showTemp(int m){
	ofstream outfile;
	outfile.open("/home/qjx/Desktop/MeshFlow_Video_Denoising-master/build/debug/QrX.txt");
	cv::Mat temp(m_size,CV_32FC1);
	temp = m_mapedX.getMat(ACCESS_READ);
	for(int i = 0;i < 400; i++){
		for (int j = 1079; j < 1080; j++){
			char *ptr = new char[10];
			ptr = gcvt(temp.at<float>(i,j), 5, ptr);
			outfile << ptr << " ";
		}
		outfile << endl;
	}
	outfile << endl;
	cout << "matric saved..." << endl;
}

void MotionDenoiser::SaveResult(char* name){
	
	cv::VideoWriter outVideoWriter;
	cout << "fps: " << m_fps << endl;
	cout << "fNum: " << m_frameNum << endl; 
	outVideoWriter.open(name, CV_FOURCC('X', 'V', 'I', 'D'),m_fps,m_size);
	for (int i = 0; i < m_frameNum; i++){
		if(i == 40)
			cv::imwrite("./wows/out20.png",Udst[i].getMat(ACCESS_READ));
		outVideoWriter << Udst[i].getMat(ACCESS_READ);
	}
	outVideoWriter.~VideoWriter();
}
#include "Mesh.h"

#ifndef __VECPT2F__
#define VecPt2f vector<cv::Point2f>
#endif __VECPT2F__

#define RADIUS 200

#ifndef __NODE__
#define __NODE__
struct node{
	VecPt2f features;
	VecPt2f motions;
};
#endif


#ifndef __MESHFLOW__
#define __MESHFLOW__
class MeshFlow{

public: int m_height,m_width;
private:
	int m_quadWidth,m_quadHeight;
	int m_meshheight,m_meshwidth;

	cv::Mat m_source,m_target;
	cv::Mat m_globalHomography;
	Mesh* m_mesh;
	Mesh* m_warpedmesh;
	VecPt2f m_vertexMotion;
	node n;
	
	cv::Mat indexX,indexY;
	cv::Mat dstX,dstY;
	cv::Mat H;
	cv::Mat X;
	cv::Mat Y;
	cv::Mat W;
	cv::Mat Tx,Ty;
private:
	void SpatialMedianFilter();
	void DistributeMotion2MeshVertexes_MedianFilter();
	void WarpMeshbyMotion();
	cv::Point2f Trans(cv::Mat &H,cv::Point2f &pt);

public:
	MeshFlow();
	void ReInitialize();
	void Initialize();
	void Execute();
	void SetFeature(vector<cv::Point2f> &spt, vector<cv::Point2f> &tpt);
	void GetMotions(cv::Mat &mapX, cv::Mat &mapY); 


	Mesh* GetDestinMesh(){return m_warpedmesh;}
	void GetWarpedSource(cv::Mat &dst,cv::Mat &mapX,cv::Mat &mapY);
};
#endif
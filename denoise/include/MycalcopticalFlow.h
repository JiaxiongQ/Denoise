//------------------------------------------------------------------------//
//                             LK-OpticalFlow related                     //
//------------------------------------------------------------------------//

#pragma once
#include "define.h"
#include <opencv2/video/tracking.hpp>
#include <iostream>
using namespace std;
namespace lkopticalflowt
{
	//------------------------------------------------------------------------//
	//                            LK-OpticalFlow                              //
	//------------------------------------------------------------------------//
#ifndef CV_DESCALE 
#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))
#endif
	namespace detail
	{

		typedef short deriv_type;

		struct LKTrackerInvoker : cv::ParallelLoopBody
		{
			LKTrackerInvoker(const cv::Mat& _prevImg, const cv::Mat& _prevDeriv, const cv::Mat& _nextImg,
				const cv::Point2f* _prevPts, cv::Point2f* _nextPts,
			uchar* _status, float* _err,
			cv::Size _winSize, cv::TermCriteria _criteria,
			int _level, int _maxLevel, int _flags, float _minEigThreshold);

			void operator()(const cv::Range& range) const;

			const cv::Mat* prevImg;
			const cv::Mat* nextImg;
			const cv::Mat* prevDeriv;
			const cv::Point2f* prevPts;
			cv::Point2f* nextPts;
			uchar* status;
			float* err;
			cv::Size winSize;
			cv::TermCriteria criteria;
			int level;
			int maxLevel;
			int flags;
			float minEigThreshold;
		};

	}// namespace detail
	void calcSharrDeriv(cv::Mat& src,
		cv::Mat& dst);
	int buildOpticalFlowPyramid(cv::InputArray img,
		cv::OutputArrayOfArrays pyramid,
		cv::Size winSize,
		int maxLevel,
		bool withDerivatives = true,
		int pyrBorder = cv::BORDER_REFLECT_101,
		int derivBorder = cv::BORDER_CONSTANT,
		bool tryReuseInputImage = true);
	void calcOpticalFlowPyrLK(cv::InputArray prevImg,
		cv::InputArray nextImg,
		cv::InputArray prevPts,
		CV_OUT cv::InputOutputArray nextPts,
		cv::OutputArray status,
		cv::OutputArray err,
		cv::Size winSize = cv::Size(21, 21),
		int maxLevel = 3,
		cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
		int flags = 0,
		double minEigThreshold = 1e-4);

}
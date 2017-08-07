#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <./boost/geometry.hpp>
#include <./boost/geometry/geometries/point_xy.hpp>
#include <./boost/geometry/geometries/polygon.hpp>

#include "../str/segmentation.h"

using namespace std;
using namespace cv;
namespace bg = boost::geometry;

typedef bg::model::d2::point_xy<double> point_type;
typedef bg::model::polygon<point_type> polygon_type;

polygon_type string2poly(string pg){
	polygon_type poly;
	bg::read_wkt(pg, poly);
	return poly;
}

vector<vector<Point>> string2cvPoint(vector<string> mps_rect) {
    vector<vector<Point>> rect;
    for(int rectindex = 0; rectindex < mps_rect.size(); ++rectindex)
    {
        vector<string> v;
        SplitString(mps_rect[rectindex], v, ",");
        vector<string> coorlt;
        vector<cv::Point> coor;
        SplitString(v[0], coorlt, " ");
        cv::Point mps_top_left_pt(atoi(coorlt[0].c_str()), atoi(coorlt[1].c_str()));
        coor.push_back(mps_top_left_pt);
        vector<string> coorrb;
        SplitString(v[1], coorrb, " ");
        cv::Point mps_bottom_right_pt(atoi(coorrb[0].c_str()), atoi(coorrb[1].c_str()));
        coor.push_back(mps_bottom_right_pt);
        rect.push_back(coor);
    }
    return rect;
}
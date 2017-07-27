//
// Created by eric on 17-7-24.
//
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;

#ifndef README_CAMERA_H_H
#define README_CAMERA_H_H

/**
 * set local camera fps by set fps_divide(not offical version)
 * ...maybe useful in somewhere(such as pc's cpu is low)
 *
 * @Params:
 * - index: if only one camera on pc, the index is 0
 * - fps_divide: the final fps = camer's origin fps/fps_divide
 */
void local_cap(int index, int fps_divide) {
    VideoCapture cap;
    Mat frame;
    vector <Mat> frames;
    bool stop = false;
    cap.open(index);
    while(!stop)
    {
        // wait for a new frame from camera and store it into 'frame'
        // read fps_divide frame into frames
        for (int i = 0; i < fps_divide; ++i) {
            cap >> frame;
            if (!frame.empty())
                frames.push_back(frame);
        }
        //just show the first img
        imshow("Live", frames[0]);
        //clear frames
        frames.clear();
        if (waitKey(30)  == 27)
            stop = true;
    }
}

#endif //README_CAMERA_H_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cv.h>
#include <highgui.h>
#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <string>
#include "./rect/calculate.h"
#include "./rect/read.h"
#include "str/segmentation.h"
#include "pic/measure.h"

using namespace cv;
using namespace std;

void test_per() {
    vector<string> mps = read("/home/eric/CLionProjects/readme/README/compile-jpg.txt");
    for (int i = 0; i < mps.size(); ++i) {
        //"POLYGON((0 0,0 200,200 200,200 0,0 0))" pg contain rect
        //"POLYGON((200 200,200 400,400 400,400 200,200 200))" pg overlap rect
        // "POLYGON((0 0,0 400,400 400,400 0,0 0))" rect contain pg
        //per("POLYGON((0 0,0 200,200 200,200 0,0 0))",  mps[i])
        // "" can't calculate pg = rect
        cout << per("POLYGON((0 0,0 200,200 200,200 0,0 0))",  mps[i]);
    }
}

void test_split() {
    vector<string> v;
    string s = "a,b,c";
    SplitString(s, v, ",");
    for (int i = 0; i < v.size(); ++i) {
        cout << v[i];
    }
    LOG(INFO) << "HELLO WORLD";
}

void test_read_img() {
    vector<Mat> imgs;
    Mat img;
    int length = 5;
    for (int i = 0; i < length; ++i) {
        img = imread("/home/eric/CLionProjects/readme/README/lena.jpg");
        imgs.push_back(img);
    }
    if (imgs[1].data) {
        namedWindow("Display Image", WINDOW_AUTOSIZE );
        imshow("Display Image", imgs[1]);
        LOG(INFO) << "success show index=1's img in imgs";
        waitKey(0);
    }
    imgs.clear();
    if(!imgs[1].data) {
        LOG(INFO) << "success clear";
    }
}

void test_mutil_video() {
    VideoCapture cap;
    Mat frame;
    bool stop = false;
    int loops = 2, fps = 5;
    vector<Mat> frames;
    cap.open(0);
    while(!stop)
    {
        // wait for a new frame from camera and store it into 'frame'
        //cap.read(frame);
        for (int i = 0; i < fps; ++i) {
            cap >> frame;
            if (!frame.empty())
                frames.push_back(frame);
        }
        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frames[0]);
        frames.clear();
        LOG(INFO) << frames.size();
        if (waitKey(30)  == 27)
            stop = true;
    }
}

int test_local_camera() {
    VideoCapture cap("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
    //VideoCapture cap("fakesrc ! videoconvert ! appsink");
    if (!cap.isOpened())
    {
        cout << "Failed to open camera." << endl;
        return -1;
    }

    for(;;)
    {
        Mat frame;
        cap >> frame;
        imshow("original", frame);
        //waitKey(1);
        if(waitKey(30) >= 0)
            break;
    }
    return 0;
    //cap.release();
}

int test_ssim() {
    // default settings
    double C1 = 6.5025, C2 = 58.5225;
    IplImage
            *img1=NULL, *img2=NULL, *img1_img2=NULL,
            *img1_temp=NULL, *img2_temp=NULL,
            *img1_sq=NULL, *img2_sq=NULL,
            *mu1=NULL, *mu2=NULL,
            *mu1_sq=NULL, *mu2_sq=NULL, *mu1_mu2=NULL,
            *sigma1_sq=NULL, *sigma2_sq=NULL, *sigma12=NULL,
            *ssim_map=NULL, *temp1=NULL, *temp2=NULL, *temp3=NULL;
    img1_temp = cvLoadImage("/home/eric/CLionProjects/readme/README/python-compare-a.jpg");
    img2_temp = cvLoadImage("/home/eric/CLionProjects/readme/README/python-compare-a.jpg");
    if(img1_temp==NULL || img2_temp==NULL)
        return -1;
    int x=img1_temp->width, y=img1_temp->height;
    int nChan=img1_temp->nChannels, d=IPL_DEPTH_32F;
    CvSize size = cvSize(x, y);
    img1 = cvCreateImage(size, d, nChan);
    img2 = cvCreateImage(size, d, nChan);

    cvConvert(img1_temp, img1);
	cvConvert(img2_temp, img2);
	cvReleaseImage(&img1_temp);
	cvReleaseImage(&img2_temp);

    img1_sq = cvCreateImage( size, d, nChan);
    img2_sq = cvCreateImage( size, d, nChan);
    img1_img2 = cvCreateImage( size, d, nChan);

    cvPow( img1, img1_sq, 2 );
    cvPow( img2, img2_sq, 2 );
    cvMul( img1, img2, img1_img2, 1 );

    mu1 = cvCreateImage( size, d, nChan);
    mu2 = cvCreateImage( size, d, nChan);

    mu1_sq = cvCreateImage( size, d, nChan);
    mu2_sq = cvCreateImage( size, d, nChan);
    mu1_mu2 = cvCreateImage( size, d, nChan);


    sigma1_sq = cvCreateImage( size, d, nChan);
    sigma2_sq = cvCreateImage( size, d, nChan);
    sigma12 = cvCreateImage( size, d, nChan);

    temp1 = cvCreateImage( size, d, nChan);
    temp2 = cvCreateImage( size, d, nChan);
    temp3 = cvCreateImage( size, d, nChan);

    ssim_map = cvCreateImage( size, d, nChan);
    /*************************** END INITS **********************************/


    //////////////////////////////////////////////////////////////////////////
    // PRELIMINARY COMPUTING
    cvSmooth( img1, mu1, CV_GAUSSIAN, 11, 11, 1.5 );
    cvSmooth( img2, mu2, CV_GAUSSIAN, 11, 11, 1.5 );

    cvPow( mu1, mu1_sq, 2 );
    cvPow( mu2, mu2_sq, 2 );
    cvMul( mu1, mu2, mu1_mu2, 1 );


    cvSmooth( img1_sq, sigma1_sq, CV_GAUSSIAN, 11, 11, 1.5 );
    cvAddWeighted( sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq );

    cvSmooth( img2_sq, sigma2_sq, CV_GAUSSIAN, 11, 11, 1.5 );
    cvAddWeighted( sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq );

    cvSmooth( img1_img2, sigma12, CV_GAUSSIAN, 11, 11, 1.5 );
    cvAddWeighted( sigma12, 1, mu1_mu2, -1, 0, sigma12 );


    //////////////////////////////////////////////////////////////////////////
    // FORMULA

    // (2*mu1_mu2 + C1)
    cvScale( mu1_mu2, temp1, 2 );
    cvAddS( temp1, cvScalarAll(C1), temp1 );

    // (2*sigma12 + C2)
    cvScale( sigma12, temp2, 2 );
    cvAddS( temp2, cvScalarAll(C2), temp2 );

    // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    cvMul( temp1, temp2, temp3, 1 );

    // (mu1_sq + mu2_sq + C1)
    cvAdd( mu1_sq, mu2_sq, temp1 );
    cvAddS( temp1, cvScalarAll(C1), temp1 );

    // (sigma1_sq + sigma2_sq + C2)
    cvAdd( sigma1_sq, sigma2_sq, temp2 );
    cvAddS( temp2, cvScalarAll(C2), temp2 );

    // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
    cvMul( temp1, temp2, temp1, 1 );

    // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
    cvDiv( temp3, temp1, ssim_map, 1 );


    CvScalar index_scalar = cvAvg( ssim_map );

    // through observation, there is approximately
    // 1% error max with the original matlab program

    cout << "(R, G & B SSIM index)" << endl ;
    cout << index_scalar.val[2] * 100 << "%" << endl ;
    cout << index_scalar.val[1] * 100 << "%" << endl ;
    cout << index_scalar.val[0] * 100 << "%" << endl ;

    // if you use this code within a program
    // don't forget to release the IplImages
    return 0;
}


int main(int argc, char* argv[]) {
    FLAGS_log_dir = "/home/eric/CLionProjects/readme/log";
    google::InitGoogleLogging(argv[0]);
    //google::SetLogDestination(google::GLOG_INFO, "./log/log_info_"); //设置 google::INFO 级别的日志存储路径和文件名前缀
    //test_per();
    //test_split();
    //test_read_img();
    //test_mutil_video();
    //test_local_camera();
    //test_ssim();
    LOG(INFO) << ssim("/home/eric/CLionProjects/readme/README/python-compare-a.jpg", "/home/eric/CLionProjects/readme/README/python-compare-a.jpg");
    return 0;
}
/************************************************
     Canny edge detection with parallelisation
************************************************/

#include <iostream>
#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>
#include <omp.h>

using namespace cv; 
using namespace std; 

int main()
{
    string filename;
    cout<<"Enter filename:";
    cin>> filename;


    // Reading image
    Mat img = imread(filename);

    // Displaying image
    //imshow("Original Image",img);
    //waitKey(0);

    // Converting to grayscale
    Mat img_gray,image_gray;
    cvtColor(img,image_gray,CV_RGB2GRAY);
    GaussianBlur( image_gray, img_gray, Size(5,5), 3, 3);

    
    // Displaying grayscale image
    //imshow("Original Image",img_gray);
    //waitKey(0);

    cv::Mat Gx, Gy; int ksize=3;
    Mat abs_grad_x, abs_grad_y;
    cv::Sobel(img_gray, Gx, CV_64F, 1, 0, ksize);
    convertScaleAbs( Gx, abs_grad_x );
    cv::Sobel(img_gray, Gy, CV_64F, 0, 1, ksize);
    convertScaleAbs( Gy, abs_grad_y );
    Mat grad;
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    //imshow("Sobel Image",grad);
    //waitKey(0);
    
   
    
    //Canny edge detection

    // Performing Non-Maximum Surpression
    float theta; // Calculate intensity gradient vector theta=atan2(Gy,Gx);
    Mat nonMaxSupp= Mat(grad.rows-2, grad.cols-2, CV_8UC1); //CV_8UC1 is 8-bit single channel image i.e grayscale
    int i,j;

   

    #pragma omp parallel for  default(shared)  private(i,j,theta) num_threads(3) ordered schedule(dynamic,120) collapse(2)
    for(i=1; i<Gx.rows-1; i++)
    {
        for(j=1; j<Gx.cols-1; j++)
        {
            //if(gradient_x.at<uchar>(i,j) ==0) //Arctan Fix
             //   theta = 90;
            //else
                theta = atan2(Gy.at<uchar>(i,j),Gx.at<uchar>(i,j))*(180/3.14);
                //theta = atan(gradient_y.at<uchar>(i,j)/gradient_x.at<uchar>(i,j))*(180/3.14);
            //cout<<theta<<endl;
            //if(theta>max)
            //    max=theta;
            nonMaxSupp.at<uchar>(i-1, j-1) = grad.at<uchar>(i,j);
            
            // For horizontal edge
            if(((-22.5 < theta) && (theta <= 22.5)) || ((157.5 < theta) && (theta <= -157.5)))
            {
                if ((grad.at<uchar>(i,j) < grad.at<uchar>(i,j+1)) || (grad.at<uchar>(i,j) < grad.at<uchar>(i,j-1)))
                    nonMaxSupp.at<uchar>(i-1, j-1) = 0;
            }

            //For vertical edge
            if (((-112.5 < theta) && (theta <= -67.5)) || ((67.5 < theta) && (theta <= 112.5)))
            {
                if ((grad.at<uchar>(i,j) < grad.at<uchar>(i+1,j)) || (grad.at<uchar>(i,j) < grad.at<uchar>(i-1,j)))
                    nonMaxSupp.at<uchar>(i-1, j-1) = 0;
            }

            // For 135 degree or -45 degree edge
            if (((-67.5 < theta) && (theta <= -22.5)) || ((112.5 < theta) && (theta <= 157.5)))
            {
                if ((grad.at<uchar>(i,j) < grad.at<uchar>(i-1,j+1)) || (grad.at<uchar>(i,j) < grad.at<uchar>(i+1,j-1)))
                    nonMaxSupp.at<uchar>(i-1, j-1) = 0;
            }

            // For 45 Degree Edge
            if (((-157.5 < theta) && (theta <= -112.5)) || ((22.5 < theta) && (theta <= 67.5)))
            {
                if ((grad.at<uchar>(i,j) < grad.at<uchar>(i+1,j+1)) || (grad.at<uchar>(i,j) < grad.at<uchar>(i-1,j-1)))
                    nonMaxSupp.at<uchar>(i-1, j-1) = 0;
            }

            /*else
            {
                nonMaxSupp.at<uchar>(i-1, j-1) = 255;    
            }
            */
        }

    }
    

    //cout<<endl<<"max"<<max;
    imshow("Non-Maximum Surpression",nonMaxSupp);
    waitKey(0);

    // Hysterisis Thresholding
    int low=20,high=40;

    if(low > 255)
        low = 255;
    if(high > 255)
        high = 255;
    
    Mat EdgeMat = Mat(nonMaxSupp.rows, nonMaxSupp.cols, nonMaxSupp.type());
    
    double start,end,diff;
    start = omp_get_wtime();

    #pragma omp parallel for  default(shared)  private(i,j) num_threads(4) ordered schedule(dynamic,120) collapse(2)
    for (i=0; i<nonMaxSupp.rows; i++) 
    {
        for (j = 0; j<nonMaxSupp.cols; j++) 
        {
            EdgeMat.at<uchar>(i,j) = nonMaxSupp.at<uchar>(i,j);
            if(EdgeMat.at<uchar>(i,j) > high)
                EdgeMat.at<uchar>(i,j) = 255;
            else if(EdgeMat.at<uchar>(i,j) < low)
                EdgeMat.at<uchar>(i,j) = 0;
            else
            {
                bool anyHigh = false;
                bool anyBetween = false;
                //for neighbouring pixels
                for (int x=i-1; x < i+2; x++) 
                {
                    for (int y = j-1; y<j+2; y++) 
                    {
                        if(x <= 0 || y <= 0 || x > EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
                            continue;
                        else
                        {
                            if(EdgeMat.at<uchar>(x,y) > high)
                            {
                                EdgeMat.at<uchar>(i,j) = 255;
                                anyHigh = true;
                                break;      
                            }
                            else if(EdgeMat.at<uchar>(x,y) <= high && EdgeMat.at<uchar>(x,y) >= low)
                                anyBetween = true;
                        }
                    }
                    if(anyHigh)
                        break;
                }
                if(!anyHigh && anyBetween)
                    for (int x=i-2; x < i+3; x++) 
                    {
                        for (int y = j-1; y<j+3; y++) 
                        {
                            if(x < 0 || y < 0 || x > EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
                                continue;
                            else
                            {
                                if(EdgeMat.at<uchar>(x,y) > high)
                                {
                                    EdgeMat.at<uchar>(i,j) = 255;
                                    anyHigh = true;
                                    break;
                                }
                            }
                        }
                        if(anyHigh)
                            break;
                    }
                if(!anyHigh)
                    EdgeMat.at<uchar>(i,j) = 0;
            }
        }
    }
    end = omp_get_wtime();
    diff = end-start;
    cout<<"\nparallel code time:"<<diff<<endl;
    
    imshow("Hysterisis Thresholding",EdgeMat);
    waitKey(0);

    return 0;

}
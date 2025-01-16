#include <iostream>
#include "common_code.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat
fsiv_compute_image_histogram(const cv::Mat &in)
{
    CV_Assert(in.type() == CV_8UC1);
    cv::Mat hist;
    // TODO
    // Hint: Use the function cv::calcHist.
    int hist_size = 256;
    float range[] = {0, 256};
    const float* hist_range = {range};
    
    //Calcular el histograma y guardarlo en hist.
    cv::calcHist(&in,1,0,cv::Mat(),hist,1,&hist_size,&hist_range,true,false);          
    //
    CV_Assert(!hist.empty());
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    return hist;
}

void fsiv_normalize_histogram(cv::Mat &hist)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);

    // TODO
    // Hint: Use the function cv::normalize() with norm L1 = 1.0
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    //

    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    CV_Assert(cv::sum(hist)[0] == 0.0 || cv::abs(cv::sum(hist)[0] - 1.0) <= 1.0e-6);
}

void fsiv_accumulate_histogram(cv::Mat &hist)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);

    // TODO
    for(int i= 1; i < hist.rows; ++i){
        hist.at<float>(i)+= hist.at<float>(i - 1);
    }
    //

    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
}

void fsiv_compute_clipped_histogram(cv::Mat &h, float cl)
{
    CV_Assert(h.type() == CV_32FC1);
    CV_Assert(h.rows == 256 && h.cols == 1);

    // TODO
    // Remember: the ouput histogram will be clipped to cl value and the
    // residual area will be redistributed equally in all the histogram's bins.
    float residual = 0.0f;

    for(int i= 0; i< h.rows; ++i){

        float bin_value = h.at<float>(i);
        if(bin_value > cl){
            residual += (bin_value - cl);
            h.at<float>(i)= cl;
        }
    }

    float redistribution_amount = residual / h.rows;

    for(int i= 0; i< h.rows; ++i){
        h.at<float>(i) += redistribution_amount;
    }
    //

    CV_Assert(h.type() == CV_32FC1);
    CV_Assert(h.rows == 256 && h.cols == 1);
}

float fsiv_compute_actual_clipping_histogram_value(const cv::Mat &h, float s)
{
    CV_Assert(h.type() == CV_32FC1);
    CV_Assert(h.rows == 256 && h.cols == 1);

    int CL = s * cv::sum(h)[0] / h.rows;

    // TODO: Code the algorithm shown in the practical assignment description.

    float bot= 0.0;
    float top= CL;
   
    while((top - bot)> 1.0){

        float mid= ((top + bot)/2.0);
        float R= 0.0;

        for(int i= 0; i< h.rows; i++){

            float bin= h.at<float>(i, 0);
            
            if(bin > mid){
                R+= (bin - mid);
            }
        }
    
        if(R<= ((CL - mid)*h.rows)){
            bot= mid;
        }
        else{
            top= mid;  
        }
    }
    
    CL= bot;
    //

    return CL;
}

cv::Mat
fsiv_create_equalization_lookup_table(const cv::Mat &hist,
                                      float s)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    cv::Mat lkt = hist.clone();

    if (s >= 1.0)
    {
        // TODO: Clip the histogram.
        // Hint: use fsiv_compute_actual_clipping_histogram_value to compute the
        //       clipping level.
        // Hint: use fsiv_compute_clipped_histogram to clip the histogram.

        float clipping_value= fsiv_compute_actual_clipping_histogram_value(hist, s);

        fsiv_compute_clipped_histogram(lkt, clipping_value);
        //
    }

    // TODO: Build the equalization transform function.
    // Remember: the transform function will be the accumulated normalized
    //           image histogram.
    // Hint: use cv::Mat::convertTo() method to convert the float range [0.0, 1.0]
    //       to [0, 255] byte range.
    //
    fsiv_normalize_histogram(lkt);
    
    fsiv_accumulate_histogram(lkt);
    
    lkt.convertTo(lkt, CV_8UC1, 255.0);
    //

    CV_Assert(lkt.type() == CV_8UC1);
    CV_Assert(lkt.rows == 256 && lkt.cols == 1);
    return lkt;
}

cv::Mat
fsiv_apply_lookup_table(const cv::Mat &in, const cv::Mat &lkt,
                        cv::Mat &out)
{
    CV_Assert(in.type() == CV_8UC1);
    CV_Assert(lkt.type() == CV_8UC1);
    CV_Assert(lkt.rows == 256 && lkt.cols == 1);
    CV_Assert(out.empty() || (out.type() == CV_8UC1 &&
                              out.rows == in.rows && out.cols == in.cols));

    // TODO
    // Hint: you can use the cv::LUT function.
    cv::LUT(in, lkt, out);
    //
    CV_Assert(out.rows == in.rows && out.cols == in.cols && out.type() == in.type());
    return out;
}

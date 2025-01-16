#include "common_code.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

cv::Mat fsiv_color_rescaling(const cv::Mat &in, const cv::Scalar &from, const cv::Scalar &to)
{
    CV_Assert(in.type() == CV_8UC3);
    cv::Mat out;
    // TODO
    // HINT: use cv:divide to compute the scaling factor.
    // HINT: use method cv::Mat::mul() to scale the input matrix.

    out= in.clone();
    
    cv::Scalar scale;
    cv::divide(to, from, scale);  
    
    std::vector<cv::Mat> canales(3);
    cv::split(out, canales);  
   
    canales[0]= canales[0].mul(scale[0]);
    canales[1]= canales[1].mul(scale[1]);
    canales[2]= canales[2].mul(scale[2]);
    
    cv::merge(canales, out);
   
    out.convertTo(out, CV_8UC3);

    //
    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

cv::Mat fsiv_gray_world_color_balance(cv::Mat const &in)
{
    CV_Assert(in.type() == CV_8UC3);
    cv::Mat out;
    // TODO
    //  HINT: use cv::mean to compute the mean pixel value.

    out= in.clone();
    
    cv::Scalar avg_color= cv::mean(in); 

    out= fsiv_color_rescaling(in, avg_color, cv::Scalar::all(128));

    //
    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

cv::Mat fsiv_convert_bgr_to_gray(const cv::Mat &img, cv::Mat &out)
{
    CV_Assert(img.channels() == 3);
    // TODO
    // HINT: use cv::cvtColor()

    cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);

    //
    CV_Assert(out.channels() == 1);
    return out;
}

cv::Mat fsiv_compute_image_histogram(cv::Mat const &img)
{
    CV_Assert(img.type() == CV_8UC1);
    cv::Mat hist;
    // TODO
    // Hint: use cv::calcHist().

    int hist_size= 256;
    float range[]= {0, 256};
    const float* hist_range = {range};

    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, true, false);

    hist.convertTo(hist, CV_32FC1);

    //
    CV_Assert(!hist.empty());
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    return hist;
}

float fsiv_compute_histogram_percentile(cv::Mat const &hist, float p_value)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    CV_Assert(0.0 <= p_value && p_value <= 1.0);

    int p = 0;

    // TODO
    // Remember: find the smaller index 'p' such that
    //           sum(h[0], h[1], ... , h[p]) >= p_value*area(hist)
    // Hint: use cv::sum() to compute the histogram area.

    float area_total= cv::sum(hist)[0];
    float sum = 0.0;

    for(p= 0; p< hist.rows; ++p){
        sum+= hist.at<float>(p, 0);

        if(sum>= (p_value * area_total)){
            break;  
        }
    }

    //

    CV_Assert(0 <= p && p < hist.rows);
    return p;
}

cv::Mat fsiv_white_patch_color_balance(cv::Mat const &in, float p)
{
    CV_Assert(in.type() == CV_8UC3);
    CV_Assert(0.0f <= p && p <= 100.0f);
    cv::Mat out;
    if (p == 0.0)
    {
        // TODO
        // HINT: convert to GRAY color space to get the illuminance.
        // HINT: use cv::minMaxLoc to locate the brightest pixel.
        // HINT: use fsiv_color_rescaling when the "from" scalar was computed.

        cv::Mat gray;

        gray= fsiv_convert_bgr_to_gray(in, gray);

        double minVal, maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(gray, &minVal, &maxVal, nullptr, &maxLoc); 

        cv::Vec3b brightest_pixel= in.at<cv::Vec3b>(maxLoc);

        out= fsiv_color_rescaling(in, cv::Scalar(brightest_pixel), cv::Scalar::all(255.0));

        //
    }
    else
    {
        // TODO
        // HINT: convert to GRAY color space to get the illuminance.
        // HINT: Compute a gray level histogram to find the 100-p percentile.
        // HINT: use operator >= to get the mask with p% brighter pixels and use it
        //        to compute the mean value.
        // HINT: use fsiv_color_rescaling when the "from" scalar was computed.

        cv::Mat gray;
        
        gray= fsiv_convert_bgr_to_gray(in, gray);

        cv::Mat hist= fsiv_compute_image_histogram(gray);

        float percentile_index= fsiv_compute_histogram_percentile(hist, (100.0f - p) / 100.0f);

        cv::Mat mask= (gray>= percentile_index);

        cv::Scalar brightest_mean= cv::mean(in, mask); 

        out= fsiv_color_rescaling(in, brightest_mean, cv::Scalar::all(255.0));

        //
    }

    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

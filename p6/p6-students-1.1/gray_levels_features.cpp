/**
 *  @file gray_levels_features.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <opencv2/imgproc.hpp>
#include "gray_levels_features.hpp"

std::string
GrayLevelsFeatures::get_extractor_name () const
{
    std::string name = "Gray levels ";
    switch (int(params_[0]))
    {
        case 0:
            name += "[0,1] normalized.";
            break;
        case 1:
            name += "mean,stddev normalized.";
            break;        
        // Add you own methods, for example:
        // case 2:
        //     name += "equalized [0, 1] normalized.";
        //     break;            

        default:
            throw std::runtime_error("unknown type of gray level extractor.");
            break;
    }
    return name;
}

cv::Mat
fsiv_extract_01_normalized_graylevels (cv::Mat const& img)
{
    cv::Mat feature;
    // TODO
    // Remember: we want gray levels. When input has three channels, 
    //  a BGR color space is assumed.
    // Hint: use cv:normalize to normalize the input to full range [0, 1].
    // Hint: use cv::Mat::reshape() method to pass from WxH to 1xW*H row vector.

    //
    CV_Assert(feature.rows==1);
    CV_Assert(feature.type()==CV_32FC1);
    return feature;
}

cv::Mat
fsiv_extract_mean_stddev_normalized_gray_levels(cv::Mat const& img)
{
    
    cv::Mat feature;
    // TODO        
    // Remember: we want gray levels. When input has three channels, 
    //  a BGR color space is assumed.
    // Hint: use cv::meanStdDev to get the image's mean and stdDev.
    // Hint: use cv::Mat::reshape() method to pass from WxH to 1xW*H row vector.

    //
    CV_Assert(feature.rows==1);
    CV_Assert(feature.type()==CV_32FC1);
    return feature;
}

GrayLevelsFeatures::GrayLevelsFeatures()
{
    type_ = FSIV_GREY_LEVELS;
    params_ = {0.0}; //by default use [0, 1] normalized gray levels.
}

GrayLevelsFeatures::~GrayLevelsFeatures() {}

cv::Mat
GrayLevelsFeatures::extract_features(const cv::Mat& img)
{    
    cv::Mat feature;
    switch (int(params_[0]))
    {
        case 0:
            feature = fsiv_extract_01_normalized_graylevels(img);
            break;
        case 1:
            feature = fsiv_extract_mean_stddev_normalized_gray_levels(img);
            break;
        default:
            throw std::runtime_error("Unknown gray level feature extractor type: " 
                + std::to_string(int(params_[0])));
            break;
    }
    CV_Assert(feature.rows==1);
    CV_Assert(feature.type()==CV_32FC1);
    return feature;
}

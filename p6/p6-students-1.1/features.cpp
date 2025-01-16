/**
 *  @file features.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <iostream>
#include <exception>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "features.hpp"

// TODO: add the include for each extractor to use.
// Hint: use gray_levels_features.hpp and gray_levels_features.cpp as model to
//   make yours.
#include "gray_levels_features.hpp"
//#include "xxxxxx.hpp"

// Remember: update CMakeLists.txt with the new files.
//

FEATURE_IDS
FeaturesExtractor::get_extractor_type() const
{
    return type_;
}

cv::Ptr<FeaturesExtractor> FeaturesExtractor::create(FEATURE_IDS id)
{
    cv::Ptr<FeaturesExtractor> extractor;
    switch (id)
    {
    case FSIV_GREY_LEVELS:
    {
        extractor = cv::makePtr<GrayLevelsFeatures>();
        break;
    }

        // TODO: add here 'cases' for your features.
        // case FSIV_XXXXX: {
        //    extractor = cv::makePtr<FeatureExtractor>(new XXXXX());
        //    break;
        // }
        //

    default:
    {
        throw std::runtime_error("Error: unknown feature id.");
        break;
    }
    }
    CV_Assert(extractor != nullptr);
    return extractor;
}

cv::Mat
fsiv_extract_features(const cv::Mat &dt,
                      cv::Ptr<FeaturesExtractor> &extractor)
{
    // Process the first image to get the features dimension.
    cv::Mat feature = extractor->extract_features(dt.row(0));

    // Allocate memory.
    cv::Mat X(dt.rows, feature.cols, CV_32F);
    feature.copyTo(X.row(0));

    // Process the rest of dt.
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 1; i < dt.rows; ++i)
        extractor->extract_features(dt.row(i)).copyTo(X.row(i));
    return X;
}

void FeaturesExtractor::set_params(const std::vector<float> &new_p)
{
    if (new_p.size() != 0)
        params_ = new_p;
}

const std::vector<float> &
FeaturesExtractor::get_params() const
{
    return params_;
}

void FeaturesExtractor::train(const cv::Mat &samples)
{
    // do nothing.
    // Override this method in your class if it is needed.
    return;
}

bool FeaturesExtractor::save_model(std::string const& model_fname) const
{
    bool ret_v = false;
    cv::FileStorage f (model_fname, cv::FileStorage::APPEND);
    if (f.isOpened())
    {
        ret_v = true;
        f << "fsiv_feature_id" << int(type_);
        f << "fsiv_feature_params" << params_;
    }
    return ret_v;
}

bool FeaturesExtractor::load_model(std::string const& model_fname)
{    
    cv::FileStorage f (model_fname, cv::FileStorage::READ);

    auto node = f["fsiv_feature_id"];
    if (node.empty() || !node.isInt())
        throw std::runtime_error("Could not load the 'fsiv_feature_id' "
                                 "label from file.");
    int loaded_type;
    node >> loaded_type;
    if (loaded_type != int(type_))
        throw std::runtime_error("Trainned model for a different "
                                 "feature extractor.");
    node = f["fsiv_feature_params"];
    if (node.empty())
        throw std::runtime_error("Could not load the 'fsiv_feature_params' "
                                 "label from file.");
    node >> params_;
    return true;
}

cv::Ptr<FeaturesExtractor>
FeaturesExtractor::create(const std::string &fname)
{
    cv::Ptr<FeaturesExtractor> extr;
    cv::FileStorage f;
    f.open(fname, cv::FileStorage::READ);
    if (f.isOpened())
    {
        auto node = f["fsiv_feature_id"];
        if (node.empty() || !node.isInt())
            throw std::runtime_error("Could not load the 'fsiv_feature_id' "
                                     "label from file.");
        int loaded_type;
        node >> loaded_type;
        extr = create(FEATURE_IDS(loaded_type));
        extr->load_model(fname);        
    }
    return extr;
}
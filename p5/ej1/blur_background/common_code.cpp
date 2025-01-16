#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "common_code.hpp"

void fsiv_compute_dense_optical_flow(cv::Mat const &prev,
                                     cv::Mat const &next,
                                     cv::Mat &flow)
{
    CV_Assert(next.type() == CV_8UC1);
    CV_Assert(prev.type() == CV_8UC1);
    CV_Assert(flow.empty() || flow.type() == CV_32FC2);

    static cv::Ptr<cv::FarnebackOpticalFlow> alg;

    // TODO
    // Hint: if alg is nullptr, first you must create a new algorithm instance
    //    using cv::FarnebackOpticalFlow::create().
    // Remember: if flow is not empty, you must use it as initial estimate
    //    setting the corresponding flag in the algorithm. If it is empty, unset
    //    this flag.
    if (alg== nullptr) {
        alg= cv::FarnebackOpticalFlow::create();
    }

    alg->calc(prev, next, flow);
    //
    CV_Assert(flow.type() == CV_32FC2);
}
//BIEN
void fsiv_compute_optical_flow_magnitude(cv::Mat &flow, cv::Mat &mag)
{
    CV_Assert(flow.type() == CV_32FC2);

    // TODO
    // Hint: use cv::magnitude.
     // Separar el flujo óptico en componentes x e y.
    std::vector<cv::Mat> flow_channels(2);
    cv::split(flow, flow_channels);

    // Calcular la magnitud de los vectores.
    cv::magnitude(flow_channels[0], flow_channels[1], mag);
    //
    CV_Assert(mag.type() == CV_32FC1);
}
//BIEN
cv::Mat
fsiv_create_structuring_element(int ste_r, int type)
{
    cv::Mat ste;
    // TODO
    // Hint: use cv::getStructuringElement.
    ste = cv::getStructuringElement(type, cv::Size(2 * ste_r + 1, 2 * ste_r + 1));
    //
    return ste;
}

void fsiv_compute_of_foreground_mask(cv::Mat const &prev, cv::Mat const &curr,
                                     cv::Mat &flow,
                                     cv::Mat &mask,
                                     const double t,
                                     const int ste_r,
                                     const int ste_type,
                                     const float alpha)
{
    CV_Assert(!prev.empty() && prev.size() == curr.size());
    CV_Assert(prev.type() == CV_8UC1 && prev.type() == curr.type());
    CV_Assert(mask.empty() || mask.size() == prev.size());
    CV_Assert(alpha >= 0.0 && alpha <= 1.0);

    // TODO
    // The steps are:
    // 1. Compute the optical flow.
    // 2. Compute the magnitude of the optical flow.
    // 3. Threshold the magnitude (>= th) to get the current mask.
    // 4. If ste_r>0, dilate the current mask. Hint: use cv::dilate()
    // 5. If alpha>0.0 (and input mask is not empty), update mask using a
    //    running average (new_mask = alpha*old_mask + (1-alpha)*current_mask).
    //    When alpha=0.0, new_mask = current_mask. Hint: use cv::addWeighted() for this.
    
    //
    CV_Assert(mask.size() == prev.size());
    CV_Assert(mask.type() == CV_8UC1);
}

void fsiv_blur_background(cv::Mat const &input,
                          cv::Mat const &fg_mask,
                          cv::Mat &output,
                          const int blur_r,
                          const int blur_type)
{
    CV_Assert(input.size() == fg_mask.size());
    CV_Assert(fg_mask.type() == CV_8UC1);

    // TODO
    // Hint: use cv::blur or cv::GaussianBlur to blur the background.
    // Hint: use cv::Mat::copyTo with mask to fuse foreground and background.
    
    //
    CV_Assert(output.type() == input.type());
    CV_Assert(output.size() == input.size());
}

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_code.hpp"

void fsiv_compute_derivate(cv::Mat const &img, cv::Mat &dx, cv::Mat &dy, int g_r,
                           int s_ap)
{
    CV_Assert(img.type() == CV_8UC1);
    // TODO
    // Remember: if g_r > 0 apply a previous Gaussian Blur operation with kernel size 2*g_r+1.
    // Hint: use Sobel operator to compute derivate.
    if(g_r> 0){
        cv::GaussianBlur(img, img, cv::Size(2 * g_r + 1, 2 * g_r + 1), 0);
    }
    cv::Sobel(img, dx, CV_32FC1, 1, 0, s_ap);
    cv::Sobel(img, dy, CV_32FC1, 0, 1, s_ap);
    //
    CV_Assert(dx.size() == img.size());
    CV_Assert(dy.size() == dx.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);
}

void fsiv_compute_gradient_magnitude(cv::Mat const &dx, cv::Mat const &dy,
                                     cv::Mat &gradient)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);

    // TODO
    // Hint: use cv::magnitude.
    cv::magnitude(dx, dy, gradient);
    //

    CV_Assert(gradient.size() == dx.size());
    CV_Assert(gradient.type() == CV_32FC1);
}

void fsiv_compute_gradient_histogram(cv::Mat const &gradient, int n_bins, cv::Mat &hist, float &max_gradient)
{
    // TODO
    // Hint: use cv::minMaxLoc to get the gradient range {0, max_gradient}
    double min_val, max_val;
    cv::minMaxLoc(gradient, &min_val, &max_val);
    max_gradient = static_cast<float>(max_val);

    std::vector<int> channels = {0};
    std::vector<float> ranges = {0.0f, max_gradient};
    std::vector<int> histSize = {n_bins};

    cv::calcHist(std::vector<cv::Mat>{gradient},channels,cv::noArray(),hist,histSize,ranges);
    //
    CV_Assert(max_gradient > 0.0);
    CV_Assert(hist.rows == n_bins);
}

int fsiv_compute_histogram_percentile(cv::Mat const &hist, float percentile)
{
    CV_Assert(percentile >= 0.0 && percentile <= 1.0);
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    int idx = -1;
    // TODO
    // Hint: use cv::sum to compute the histogram area.
    // Remember: The percentile p is the first i that sum{h[0], h[1], ..., h[i]} >= p
    float total_sum = cv::sum(hist)[0];
    float cumulative_sum = 0.0f;
    

    for (int i = 0; i < hist.rows; ++i) {
        cumulative_sum += hist.at<float>(i);
        if (cumulative_sum / total_sum >= percentile) {
            idx = i;
            break;
        }
    }

    if (idx == hist.rows) {
        idx = hist.rows - 1;
    }

    //
    CV_Assert(idx >= 0 && idx < hist.rows);
    CV_Assert(idx == 0 || cv::sum(hist(cv::Range(0, idx), cv::Range::all()))[0] / cv::sum(hist)[0] < percentile);
    CV_Assert(cv::sum(hist(cv::Range(0, idx + 1), cv::Range::all()))[0] / cv::sum(hist)[0] >= percentile);
    return idx;
}

float fsiv_histogram_idx_to_value(int idx, int n_bins, float max_value,
                                  float min_value)
{
    CV_Assert(idx >= 0);
    CV_Assert(idx < n_bins);
    float value = 0.0;
    // TODO
    // Remember: Map integer range [0, n_bins) into float
    // range [min_value, max_value)
    value= min_value + (max_value - min_value) * (static_cast<float>(idx) / n_bins);
    //
    CV_Assert(value >= min_value);
    CV_Assert(value < max_value);
    return value;
}

void fsiv_percentile_edge_detector(cv::Mat const &gradient, cv::Mat &edges,
                                   float th, int n_bins)
{
    CV_Assert(gradient.type() == CV_32FC1);

    // TODO
    // Remember: user other fsiv_xxx to compute histogram and percentiles.
    // Remember: map histogram range {0, ..., n_bins} to the gradient range
    // {0.0, ..., max_grad}
    // Hint: use "operator >=" to threshold the gradient magnitude image.
    // Creamos un histograma para almacenar la distribución de los gradientes.
    cv::Mat hist;
    float max_gradient;
    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_gradient);


    // Calcula el índice del histograma que corresponde con el percentil dado.
    int percentile_idx = fsiv_compute_histogram_percentile(hist, th);

    // Convierte el índice del histograma en un valor de gradiente usando los límites del histograma.
    float threshold_value = fsiv_histogram_idx_to_value(percentile_idx, n_bins, max_gradient);

    // Umbraliza el gradiente para obtener los bordes.
    edges = cv::Mat::zeros(gradient.size(), CV_8UC1);  // Asegúrate de que edges se inicializa con tipo CV_8UC1.

    // Para cada píxel en la imagen de gradiente, si su valor es mayor o igual al umbral, es un borde.
    for (int r = 0; r < gradient.rows; ++r){
        for (int c = 0; c < gradient.cols; ++c){
            if (gradient.at<float>(r, c) >= threshold_value){
                edges.at<uchar>(r, c) = 255;  // Marca los bordes como 255 (blanco).
            }
        }
    }
    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_otsu_edge_detector(cv::Mat const &gradient, cv::Mat &edges)
{
    CV_Assert(gradient.type() == CV_32FC1);

    // TODO
    // Hint: normalize input gradient into rango [0, 255] to use
    // cv::threshold properly.
    //
    cv::Mat norm_gradient;
    gradient.convertTo(norm_gradient, CV_8UC1, 255.0 / cv::norm(gradient, cv::NORM_INF));

    cv::threshold(norm_gradient, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_canny_edge_detector(cv::Mat const &dx, cv::Mat const &dy, cv::Mat &edges,
                              float th1, float th2, int n_bins)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(th1 < th2);

    // TODO
    // Hint: convert the intput derivatives to CV_16C1 to be used with canny.
    // Remember: th1 and th2 are given as percentiles so you must transform to
    //           gradient range to be used in canny method.
    // Remember: we compute gradients with L2_NORM so we must indicate this in
    //           the canny method too.
    cv::Mat gradient;
    fsiv_compute_gradient_magnitude(dx, dy, gradient);

    cv::Mat hist;
    float max_gradient;
    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_gradient);


    int idx_low= fsiv_compute_histogram_percentile(hist, th1);
    int idx_high= fsiv_compute_histogram_percentile(hist, th2);

    float low_th= fsiv_histogram_idx_to_value(idx_low, n_bins, max_gradient);
    float high_th= fsiv_histogram_idx_to_value(idx_high, n_bins, max_gradient);

    cv::Mat dx_16s, dy_16s;
    dx.convertTo(dx_16s, CV_16SC1);
    dy.convertTo(dy_16s, CV_16SC1);

    cv::Canny(dx_16s, dy_16s, edges, low_th, high_th, true);
    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == dx.size());
}

void fsiv_compute_ground_truth_image(cv::Mat const &consensus_img,
                                     float min_consensus, cv::Mat &gt)
{
    //! TODO
    // Hint: use cv::normalize to normalize consensus_img into range (0, 100)
    // Hint: use "operator >=" to threshold the consensus image.
    cv::Mat normalized;
    cv::normalize(consensus_img, normalized, 0, 100, cv::NORM_MINMAX);

    gt= (normalized >= min_consensus);
    //
    CV_Assert(consensus_img.size() == gt.size());
    CV_Assert(gt.type() == CV_8UC1);
}

void fsiv_compute_confusion_matrix(cv::Mat const &gt, cv::Mat const &pred, cv::Mat &cm)
{
    CV_Assert(gt.type() == CV_8UC1);
    CV_Assert(pred.type() == CV_8UC1);
    CV_Assert(gt.size() == pred.size());

    // TODO
    // Remember: a edge detector confusion matrix is a 2x2 matrix where the
    // rows are ground truth {Positive: "is edge", Negative: "is not edge"} and
    // the columns are the predictions labels {"is edge", "is not edge"}
    // A pixel value means edge if it is <> 0, else is a "not edge" pixel.
    cm= cv::Mat::zeros(2, 2, CV_32FC1);

    for (int y= 0; y < gt.rows; ++y) {
        for (int x= 0; x < gt.cols; ++x) {
            bool is_edge_gt= gt.at<uint8_t>(y, x) != 0;
            bool is_edge_pred= pred.at<uint8_t>(y, x) != 0;

            if (is_edge_gt && is_edge_pred) {
                cm.at<float>(0, 0)++; // True positive
            } else if (is_edge_gt && !is_edge_pred) {
                cm.at<float>(0, 1)++; // False negative
            } else if (!is_edge_gt && is_edge_pred) {
                cm.at<float>(1, 0)++; // False positive
            } else {
                cm.at<float>(1, 1)++; // True negative
            }
        }
    }
    //
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cv::abs(cv::sum(cm)[0] - (gt.rows * gt.cols)) < 1.0e-6);
}

float fsiv_compute_sensitivity(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float sensitivity = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix
    float TP= cm.at<float>(0, 0);
    float FN= cm.at<float>(0, 1);
    sensitivity= TP / (TP + FN);
    //
    return sensitivity;
}

float fsiv_compute_precision(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float precision = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix
    float TP= cm.at<float>(0, 0);
    float FP= cm.at<float>(1, 0);
    precision= TP / (TP + FP);
    //
    return precision;
}

float fsiv_compute_F1_score(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float F1 = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix
    float precision= fsiv_compute_precision(cm);
    float sensitivity= fsiv_compute_sensitivity(cm);
    F1= 2 * (precision * sensitivity) / (precision + sensitivity);
    //
    return F1;
}

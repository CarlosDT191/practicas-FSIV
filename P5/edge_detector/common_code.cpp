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
    // Convertir la imagen a tipo flotante para realizar operaciones precisas.
    cv::Mat blurred_img;
    
    // Aplica el desenfoque Gaussiano si g_r > 0
    if (g_r > 0) {
        int kernel_size = 2 * g_r + 1; // Tamaño del kernel
        cv::GaussianBlur(img, blurred_img, cv::Size(kernel_size, kernel_size), 0);
    } else {
        blurred_img = img;
    }

    // Usa Sobel para calcular las derivadas en x y en y
    cv::Sobel(blurred_img, dx, CV_32F, 1, 0, s_ap); // Derivada en x
    cv::Sobel(blurred_img, dy, CV_32F, 0, 1, s_ap); // Derivada en y
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
    // Calcula la magnitud del gradiente usando cv::magnitude
    cv::magnitude(dx, dy, gradient);
    //

    CV_Assert(gradient.size() == dx.size());
    CV_Assert(gradient.type() == CV_32FC1);
}

void fsiv_compute_gradient_histogram(cv::Mat const &gradient, int n_bins, cv::Mat &hist, float &max_gradient)
{
    // TODO
    // Hint: use cv::minMaxLoc to get the gradient range {0, max_gradient}
    // Obtener el valor máximo del gradiente.
    double min_val, max_val;
    cv::minMaxLoc(gradient, &min_val, &max_val);
    max_gradient = static_cast<float>(max_val);

    // Comprobar que max_gradient sea mayor a cero.
    CV_Assert(max_gradient > 0.0);

    // Crear el histograma con n_bins.
    hist = cv::Mat::zeros(n_bins, 1, CV_32F);

    // Normalizar los valores del gradiente para que estén en el rango [0, n_bins).
    for (int row = 0; row < gradient.rows; ++row)
    {
        const float *grad_ptr = gradient.ptr<float>(row);
        for (int col = 0; col < gradient.cols; ++col)
        {
            float value = grad_ptr[col];
            int bin_idx = static_cast<int>((value / max_gradient) * n_bins);
            // Corregir el índice para que esté dentro del rango válido.
            bin_idx = std::min(bin_idx, n_bins - 1);
            hist.at<float>(bin_idx) += 1.0f;
        }
    }

    // Normalizar el histograma para que sea una distribución de probabilidad.
    hist /= static_cast<float>(gradient.rows * gradient.cols);
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
    // TODO
    // Hint: use cv::sum to compute the histogram area.
    // Remember: The percentile p is the first i that sum{h[0], h[1], ..., h[i]} >= p

    float total_sum = cv::sum(hist)[0];
    CV_Assert(total_sum > 0); // Ensure the histogram has some data

    if (percentile == 1.0f) {
        idx = hist.rows - 1; // Handle the 100% percentile case directly
    } else {
        float cumulative_sum = 0.0f;
        for (int i = 0; i < hist.rows; i++) {
            cumulative_sum += hist.at<float>(i, 0);
            if (cumulative_sum / total_sum >= percentile) {
                idx = i;
                break;
            }
        }
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
    float range = max_value - min_value;
    value = min_value + (static_cast<float>(idx) / n_bins) * range;
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
    cv::Mat hist;
    float max_grad;

    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_grad);
    int idx_th = fsiv_compute_histogram_percentile(hist, th);
    float th_value = fsiv_histogram_idx_to_value(idx_th, n_bins, max_grad, 0.0f);

    edges.create(gradient.size(), CV_8UC1);
    edges = cv::Scalar(0);

    for (int i = 0; i < gradient.rows; i++) {
        for (int j = 0; j < gradient.cols; j++) {
            if (gradient.at<float>(i, j) >= th_value) {
                edges.at<uchar>(i, j) = 255;
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
    double minVal, maxVal;
    cv::minMaxLoc(gradient, &minVal, &maxVal);

    cv::Mat normalized_gradient;
    gradient.convertTo(normalized_gradient, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    cv::threshold(normalized_gradient, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

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
     // Calcular la magnitud del gradiente a partir de dx y dy.
    cv::Mat gradient;
    fsiv_compute_gradient_magnitude(dx, dy, gradient);

    // Crear el histograma del gradiente y obtener el valor máximo.
    cv::Mat hist;
    float max_grad;
    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_grad);

    // Convertir los percentiles a valores de umbral en el rango de gradientes.
    int idx_th1 = fsiv_compute_histogram_percentile(hist, th1);
    int idx_th2 = fsiv_compute_histogram_percentile(hist, th2);
    float th1_value = fsiv_histogram_idx_to_value(idx_th1, n_bins, max_grad, 0.0f);
    float th2_value = fsiv_histogram_idx_to_value(idx_th2, n_bins, max_grad, 0.0f);

    // Convertir dx y dy a CV_16SC1 para usarlos con Canny.
    cv::Mat dx_16s, dy_16s;
    dx.convertTo(dx_16s, CV_16SC1);
    dy.convertTo(dy_16s, CV_16SC1);

    // Aplicar el detector de bordes Canny.
    cv::Canny(dx_16s, dy_16s, edges, th1_value, th2_value, true); // L2 norm.
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
    //! Normalizar la imagen de consenso al rango [0, 100].
    CV_Assert(consensus_img.type() == CV_32F || consensus_img.type() == CV_64F);
    
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
    // Crear una matriz de confusión 2x2 inicializada a ceros.
    cm = cv::Mat::zeros(2, 2, CV_32FC1);

    for (int i = 0; i < gt.rows; ++i)
    {
        const uchar *gt_row = gt.ptr<uchar>(i);
        const uchar *pred_row = pred.ptr<uchar>(i);
        for (int j = 0; j < gt.cols; ++j)
        {
            bool is_gt_edge = gt_row[j] != 0;   // Verdadero si es borde en el ground truth.
            bool is_pred_edge = pred_row[j] != 0; // Verdadero si es borde en la predicción.

            if (is_gt_edge && is_pred_edge)
                cm.at<float>(0, 0) += 1.0f; // True Positive (TP).
            else if (is_gt_edge && !is_pred_edge)
                cm.at<float>(0, 1) += 1.0f; // False Negative (FN).
            else if (!is_gt_edge && is_pred_edge)
                cm.at<float>(1, 0) += 1.0f; // False Positive (FP).
            else
                cm.at<float>(1, 1) += 1.0f; // True Negative (TN).
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

    // Obtener los valores de la matriz de confusión.
    float TP = cm.at<float>(0, 0); // True Positive.
    float FN = cm.at<float>(0, 1); // False Negative.

    // Calcular la sensibilidad: Sensitivity = TP / (TP + FN).
    if (TP + FN > 0)
    {
        sensitivity = TP / (TP + FN);
    }
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
    // Obtener los valores de la matriz de confusión.
    float TP = cm.at<float>(0, 0); // True Positive.
    float FP = cm.at<float>(1, 0); // False Positive.

    // Calcular la precisión: Precision = TP / (TP + FP).
    if (TP + FP > 0)
    {
        precision = TP / (TP + FP);
    }
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
    // Obtener los valores de la matriz de confusión.
    float TP = cm.at<float>(0, 0); // True Positive.
    float FP = cm.at<float>(1, 0); // False Positive.
    float FN = cm.at<float>(0, 1); // False Negative.

    // Calcular precisión y sensibilidad (recall).
    float precision = (TP + FP > 0) ? (TP / (TP + FP)) : 0.0f;
    float recall = (TP + FN > 0) ? (TP / (TP + FN)) : 0.0f;

    // Calcular el F1 score: F1 = 2 * (precision * recall) / (precision + recall).
    if (precision + recall > 0)
    {
        F1 = 2.0f * (precision * recall) / (precision + recall);
    }
    //
    return F1;
}

/*!
 This program applies a trained model to test samples and computes its accuracy.

 2024: UPTATED by Rafael Berral

*/

#include <iostream>
#include <sstream>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include "common_code.hpp"

#ifndef NDEBUG
int __Debug_Level = 0;
#endif

const char *keys =
    "{help h usage ? |      | print this message   }"
    "{t              |      | Only get test labels (no metrics), used for final upload.}"
#ifndef NDEBUG
    "{verbose        |0     | Set the verbose level.}"
#endif
    "{@dataset_path  |<none>| Dataset pathname.}"
    "{@model         |<none>| Model filename to test.}";

int main(int argc, char *const *argv)
{
  int retCode = EXIT_SUCCESS;

  try
  {

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Test a classifier using the Fashion MNIST dataset.");
    if (parser.has("help"))
    {
      parser.printMessage();
      return 0;
    }

#ifndef NDEBUG
    __Debug_Level = parser.get<int>("verbose");
#endif
    std::string dataset_path = parser.get<std::string>("@dataset_path");
    std::string model_fname = parser.get<std::string>("@model");
    bool only_test = parser.has("t");
    if (!parser.check())
    {
      parser.printErrors();
      return 0;
    }

    std::cout.setf(std::ios::unitbuf);
    cv::Mat X, y;

    fsiv_load_dataset(dataset_path, X, y, only_test);

    std::cout << "Loaded dataset with " << X.rows << " samples."
              << std::endl;

    std::cout << "Test partition with " << X.rows << " samples."
              << std::endl;

    std::cout << std::endl;
    std::cout << "Extracting features ... ";
    auto extractor = FeaturesExtractor::create(model_fname);
    std::cout << "Extracting features ... " << std::endl;
    std::cout << "Feature extractor: " << extractor->get_extractor_name()
              << std::endl;
    std::cout << "Feature extractor params: " << extractor->get_params()
              << std::endl;
    X = fsiv_extract_features(X, extractor);

    std::cout << "done." << std::endl;
    std::cout << "Extracted features use "
              << ((X.rows * X.cols * X.elemSize()) / (1024 * 1024))
              << " Mb. of memory." << std::endl;

    cv::Ptr<cv::ml::StatModel> clsf = fsiv_load_classifier_model(model_fname);

    if (clsf == nullptr || !clsf->isTrained())
    {
      std::cerr << "Error: I need a trained model!" << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << std::endl;
    std::cout << "Computing predictions ... ";
    cv::Mat predict_labels = fsiv_predict_labels(clsf, X);
    std::cout << "done.\n"
              << std::endl;
    fsiv_save_predictions(dataset_path, predict_labels);

    if (only_test == false)
    {

      std::cout << "Computing metrics ... ";
      cv::Mat cmat = fsiv_compute_confusion_matrix(y, predict_labels, 15);
      float acc = fsiv_compute_accuracy(cmat);
      cv::Mat RRs = fsiv_compute_recognition_rates(cmat);
      float m_rr = fsiv_compute_mean_recognition_rate(RRs);
      std::cout << "done.\n"
                << std::endl;

      std::cout << std::endl;
      std::cout << "Model metrics #########################\n"
                << std::endl;
      std::cout << "RR:\t";
      for (int i = 0; i < RRs.rows; ++i)
        std::cout << "('" << fsiv_get_dataset_label_name(i)
                  << "':" << RRs.at<float>(i) << ") ";
      std::cout << std::endl;
      std::cout << "mRR:\t" << m_rr << std::endl;
      std::cout << "Acc:\t" << acc << std::endl;
      size_t model_size = 0;
      if (fsiv_compute_file_size(model_fname, model_size))
      {
        float model_size_mb = model_size / (1024.0 * 1024.0);
        std::cout << "Model size: " << model_size_mb << " Mb." << std::endl;
        float size_score = std::max(0.0, 1.0 - (model_size_mb / (4.0 * 45.06)));
        std::cout << "Size score max(0.0, 1.0-(model_size_mb/dataset_size_mb)) = "
                  << size_score << std::endl;
        //std::cout << "Test final score 2*(acc*size_score)/(acc+size_score) = "
        //          << (2.0 * acc * size_score) / (acc + size_score) << std::endl;
      }
      else
        throw std::runtime_error("Error: could not open the file " + model_fname);
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}

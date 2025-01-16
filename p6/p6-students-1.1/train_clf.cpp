/*!
 This program trains a new model from the provided data samples.

 2024: UPTATED by Rafael Berral

*/

#include <iostream>
#include <sstream>
#include <exception>
#include <time.h>
#include <stdlib.h>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "common_code.hpp"

#ifndef NDEBUG
int __Debug_Level = 0;
#endif

const char* keys =
    "{help h usage ? |      | print this message   }"
    "{rseed        |0     | Use this value as random seed. Default 0 means use time(0)}"
    "{s_ratio      |0.5   | Use a subsample ratio size of the dataset. Default 50% of the dataset.}"
    "{f            |1     | Feature to extract. Default 1 is normalized gray levels. f_params=0 means [0,1] normalized."
                            " f_params=1 means mean/stddev normalized.}"
    "{f_params     |0     | Feature extractor parameters (if any). Format <value>[:<value>:<value>...].}"
    "{v validate   |0.1     | Use the (v*100)% of the dataset to validate."
                             "and validate. Default is to use 10% of samples to validate.}"
    "{clf          |0     | Classifier to train/test. 0: K-NN, 1:SVM, 2:RTREES.}"
    "{knn_K        |1     | Parameter K for K-NN class.}"
    "{svm_C        |1.0   | Parameter C for SVM class.}"
    "{svm_K        |0     | Kernel to use with SVM class. 0:Linear, 1:Polynomial. "
    "2:RBF, 3:SIGMOID, 4:CHI2, 5:INTER}"
    "{svm_D        |3.0   | Degree of svm polynomial kernel.}"
    "{svm_G        |1.0   | Gamma for svm RBF kernel.}"
    "{rtrees_V     |0     | Num of random features sampled per node. "
    "Default 0 meas sqrt(num. of total features).}"
    "{rtrees_T     |50    | Max num. of rtrees in the forest.}"
    "{rtrees_E     |0.1   | OOB error to stop adding more rtrees.}"
    "{@train_path  |<none>| Train dataset pathname.}"
    "{@valid_path  |<none>| Validation dataset pathname.}"
    "{@test_path   |<none>| Test dataset pathname.}"
    "{@model         |<none>| Model filename to save/load.}"
#ifndef NDEBUG
    "{verbose        |0     | Set the verbose level.}"
#endif
    ;

std::vector<float>
parse_feature_params(const std::string& f_params)
{
    std::vector<float> feature_params;
    std::istringstream in (f_params);
    float v;
    while (in)
    {
        in >> v;
        if (in)
            feature_params.push_back(v);
    }
    return feature_params;
}

int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;

  try {

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Train a classifier using the BAA500 pollen dataset.");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }

#ifndef NDEBUG
      __Debug_Level = parser.get<int>("verbose");
#endif
      FEATURE_IDS feature_id = FEATURE_IDS(parser.get<int>("f"));
      std::vector<float> feature_params =
              parse_feature_params(parser.get<std::string>("f_params"));
      float validate = parser.get<float>("v");

      std::string train_path = parser.get<std::string>("@train_path");
      std::string valid_path = parser.get<std::string>("@valid_path");
      std::string test_path = parser.get<std::string>("@test_path");

      std::string model_fname = parser.get<std::string>("@model");
      int classifier = parser.get<int>("clf");
      int knn_K = parser.get<int>("knn_K");
      float svm_C = parser.get<float>("svm_C");
      int svm_K = parser.get<int>("svm_K");
      float svm_D = parser.get<float>("svm_D");
      float svm_G = parser.get<float>("svm_G");
      int rtrees_V = parser.get<int>("rtrees_V");
      int rtrees_T = parser.get<int>("rtrees_T");
      double rtrees_E = parser.get<double>("rtrees_E");
      float s_ratio = parser.get<float>("s_ratio");
      size_t seed = parser.get<size_t>("rseed");
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

      std::cout.setf(std::ios::unitbuf);

      if (seed==0)
        seed = time(0);
      std::cout << "Set the random seed to: " << seed << std::endl;
      cv::theRNG().state = seed;

      cv::Mat X_t, y_t, X_v, y_v;
      cv::Mat X_s, y_s;

      fsiv_load_dataset(train_path, X_t, y_t);
      fsiv_subsample_dataset(X_t, y_t, X_s, y_s, s_ratio);

      X_t = X_s;
      y_t = y_s;

      fsiv_load_dataset(valid_path, X_v, y_v);

      std::cout << "Train partition with " << X_t.rows << " samples."
                << std::endl;

      if (validate>0)
        std::cout << "Validation partition with "
                    << (X_v.empty()?0:X_v.rows)
                    << " samples." << std::endl;
      std::cout << std::endl;

      auto extractor = FeaturesExtractor::create(feature_id);
      extractor->set_params(feature_params);   
      std::cout << "Feature extractor: " << extractor->get_extractor_name()
                << std::endl;
      std::cout << "Feature extractor params: " << extractor->get_params()
                << std::endl;
      std::cout << "Training feature extractor ... " << std::endl;
      extractor->train(X_t);
      std::cout << "Done." << std::endl;
      std::cout << "Extracting features ... " << std::endl;
      X_t = fsiv_extract_features(X_t, extractor);
      if (!X_v.empty())
          X_v = fsiv_extract_features(X_v, extractor);

      std::cout << "done." << std::endl;
      std::cout << "Extracted features use " <<
                   ((X_t.rows*X_t.cols*X_t.elemSize())+
                   (X_v.empty() ? 0 : ((X_v.rows*X_v.cols*X_v.elemSize()))))/(1024*1024)
                << " Mb of memory." << std::endl;
      std::cout << std::endl;

      cv::Ptr<cv::ml::StatModel> clsf;
      if (classifier == 0)
      {
        std::cout << "Using a K-NN classifier with k=" << knn_K << std::endl;
          clsf = fsiv_create_knn_classifier(knn_K);
      }
      else if (classifier == 1)
      {
        std::cout << "Using a SVM classifier with K=" << svm_K
                      << " C=" << svm_C << " D=" << svm_D << " G=" << svm_G
                      << std::endl;
          clsf = fsiv_create_svm_classifier(svm_K, svm_C, svm_D, svm_G);
      }
      else if (classifier == 2)
      {
        std::cout << "Using a RTrees classifier with V=" << rtrees_V
                      << " T=" << rtrees_T << " E=" << rtrees_E
                      << std::endl;
          clsf = fsiv_create_rtrees_classifier(rtrees_V, rtrees_T, rtrees_E);
      }
      else
      {
          std::cerr << "Error: unknown classifier." << std::endl;
          return EXIT_FAILURE;
      }


      std::cout << std::endl;
      std::cout << "Training ... ";
      fsiv_train_classifier(clsf, X_t, y_t);      
      std::cout << "done." << std::endl;


      std::cout << "Computing training accuracy ... ";
      cv::Mat predict_labels = fsiv_predict_labels(clsf, X_t);
      cv::Mat cmat = fsiv_compute_confusion_matrix(y_t, predict_labels, 15);
      float acc = fsiv_compute_accuracy(cmat);      
      std::cout << "done." << std::endl;
      std::cout << "Training accuracy: " << acc << std::endl;
      std::cout << std::endl;

      if (validate>0.0)
      {
          std::cout << "Validating ... ";
          predict_labels = fsiv_predict_labels(clsf, X_v);
          std::cout << "done." << std::endl;
          cmat = fsiv_compute_confusion_matrix(y_v, predict_labels, 15);
          acc = fsiv_compute_accuracy(cmat);
          std::cout << "Validation accuracy: " << acc << std::endl;
          std::cout << std::endl;
      }

      std::cout << "Saving the model to '" << model_fname << "'." << std::endl;
      
      // First, save the classifier's model.
      fsiv_save_classifier_model(clsf, model_fname);

      // Second, save the feature extractor model.
      extractor->save_model(model_fname);
      cv::FileStorage fs(model_fname, cv::FileStorage::APPEND);
      fs << "fsiv_random_seed" << static_cast<double>(seed);
      fs.release();

      // Third, compute model size.
      size_t model_size = 0;
      if (fsiv_compute_file_size(model_fname, model_size))
      { 
        float model_size_mb = model_size/(1024.0*1024.0);
        std::cout << "Model size: " << model_size_mb << " Mb." << std::endl;
        float size_score = std::max(0.0, 1.0-(model_size_mb/(4.0*45.06)));
        std::cout << "Size score max(0.0, 1.0-(model_size_mb/dataset_size_mb)) = "
          << size_score << std::endl;
        std::cout << "Predicted final score 2*(acc*size_score)/(acc+size_score) = " 
          <<  (2.0*acc*size_score)/(acc+size_score) << std::endl;
      }
      else
        throw std::runtime_error("Error: could not open the file " + model_fname);                  
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}

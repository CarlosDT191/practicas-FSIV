#include <iostream>
#include <exception>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset.hpp"

static cv::Mat
to_Mat(std::vector<std::vector<std::uint8_t>> const &images)
{
    cv::Mat mat = cv::Mat(images.size(), 16384 /*128x128*/, CV_8UC1);
    for (size_t i = 0; i < images.size(); i++)
        std::copy(images[i].begin(), images[i].end(), mat.row(i).begin<uint8_t>());
    return mat;
}

static cv::Mat
to_Mat(std::vector<std::uint8_t> const &labels)
{
    cv::Mat mat = cv::Mat(labels.size(), 1, CV_8UC1);
    std::copy(labels.begin(), labels.end(), mat.begin<uint8_t>());
    return mat;
}

void fsiv_load_dataset(std::string &folder,
                       cv::Mat &X, cv::Mat &y, bool ignore_labels)
{
    // Images and labels
    std::vector<std::vector<std::uint8_t>> images;
    std::vector<std::uint8_t> labels;

    // Label file
    std::string labels_csv = folder + ".csv";

    // Load label file
    std::ifstream label_file(labels_csv);
    std::string line;

    std::getline(label_file, line);

    // Load images/labels line by line
    while (std::getline(label_file, line))
    {
        std::stringstream line_stream(line);
        std::string image_filename;
        std::string label;

        if (std::getline(line_stream, image_filename, ',') && line_stream >> label)
        {
            std::string image_path = folder + "/" + image_filename;
            cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            std::vector<std::uint8_t> img_vec(img.data, img.data + img.total() * img.elemSize());

            if (!img.empty())
            {
                images.push_back(img_vec);
                if (ignore_labels == false)
                {
                    labels.push_back(fsiv_get_dataset_label_id(label));
                }
                else
                {
                    labels.push_back(15);
                }
            }
        }
    }

    label_file.close();

    X = to_Mat(images);
    y = cv::Mat_<std::int32_t>(to_Mat(labels));

    CV_Assert(X.rows == y.rows);
    CV_Assert(X.type() == CV_8UC1);
    CV_Assert(y.type() == CV_32SC1);
}

static std::string fsiv_pollen_label_names[] = {"alnus", "betula",
                                                "carpinus", "corylus", "cupressaceae", "fagus", "fraxinus", "picea", "pinus",
                                                "poaceae", "populus", "quercus", "salix", "tilia", "urticaceae"};

const std::string &
fsiv_get_dataset_label_name(int id)
{
    CV_Assert(0 <= id && id < 15);
    return fsiv_pollen_label_names[id];
}

const int
fsiv_get_dataset_label_id(std::string &name)
{
    static std::map<std::string, int> fsiv_pollen_label_map;
    static bool is_map_populated = false;

    if (is_map_populated == false)
    {
        for (int i = 0; i < 15; i++)
        {
            fsiv_pollen_label_map[fsiv_pollen_label_names[i]] = i;
        }
        is_map_populated = true;
    }

    CV_Assert(fsiv_pollen_label_map.count(name) == 1);
    return fsiv_pollen_label_map[name];
}

void fsiv_split_dataset(float val_percent, const cv::Mat &X,
                        const cv::Mat &y,
                        cv::Mat &X_t, cv::Mat &y_t,
                        cv::Mat &X_v, cv::Mat &y_v)
{
    CV_Assert(0.0 <= val_percent && val_percent < 1.0);
    int train_size = X.rows * (1.0 - val_percent);
    X_t = X.rowRange(0, train_size);
    y_t = y.rowRange(0, train_size);
    X_v = X.rowRange(train_size, X.rows);
    y_v = y.rowRange(train_size, y.rows);
    CV_Assert(X.rows == (X_t.rows + X_v.rows));
    CV_Assert(y.rows == (y_t.rows + y_v.rows));
}

bool fsiv_compute_file_size(std::string const &path, size_t &size)
{
    bool success = true;
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (file)
        size = file.tellg();
    else
        success = false;

    return success;
}

void fsiv_subsample_dataset(const cv::Mat &X, const cv::Mat &y,
                            cv::Mat &X_s, cv::Mat &y_s, float p)
{
    CV_Assert(X.rows == y.rows);
    CV_Assert(p > 0.0 && p <= 1.0f);
    const int subsample_size = X.rows * p;
    cv::Mat idxs = cv::Mat(X.rows, 1, CV_32SC1);
    for (int i = 0; i < X.rows; ++i)
        idxs.at<int>(i) = i;
    cv::randShuffle(idxs);
    X_s = cv::Mat(subsample_size, X.cols, X.type());
    y_s = cv::Mat(subsample_size, y.cols, y.type());

#ifdef USE_OPENMP
#pragma openmp parallel for
#endif
    for (int i = 0; i < subsample_size; ++i)
    {
        const int idx = idxs.at<int>(i);
        X.row(idx).copyTo(X_s.row(i));
        y.row(idx).copyTo(y_s.row(i));
    }
}

void fsiv_save_predictions(std::string &path, cv::Mat &y){
    std::string labels_csv = path + ".csv";
    std::string predicted_csv = path + "_predicted.csv";

    std::ifstream label_file(labels_csv);
    std::ofstream predicted_file(predicted_csv);
    std::string line;

    std::getline(label_file, line);
    predicted_file << line << "\n";

    // Load images/labels line by line
    int i = 0;
    std::vector<std::int32_t> label_vector(y.begin<std::int32_t>(), y.end<std::int32_t>());
    while (std::getline(label_file, line))
    {
        std::stringstream line_stream(line);
        std::string image_filename;
        std::string label;

        if (std::getline(line_stream, image_filename, ',') && line_stream >> label)
        {
            predicted_file << image_filename << "," << fsiv_get_dataset_label_name(label_vector[i]) << "\n";
            i = i + 1;
        }
    }

    label_file.close();
    predicted_file.close();
}
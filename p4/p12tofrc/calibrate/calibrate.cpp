/**
 * @file calibrate.cpp
 * @author Francisco José Madrid Cuevas (fjmadrid@uco.es)
 * @brief Calibrate the intrinsic parameters of a camera.
 * @version 1.5
 * @date 2024-09-24
 *
 * @copyright (C) Copyright 2024- This work is openly licensed via CC-BY-NC-SA 4.0. See more details here: https://creativecommons.org/licenses/by-nc-sa/4.0/
 *
 */
#include <iostream>
#include <iomanip>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "common_code.hpp"

const cv::String keys =
    "{help h usage ? |      | print this message.}"
    "{verbose        |      | activate verbose mode.}"
    "{c              |      | The input is a camera. Specify the camera index as input_1.}"
    "{c_width        |640   | The camera width.}"
    "{c_height       |480   | The camera height.}"
    "{v              |      | The input is a video file. Specify the video file pathname as input_1.}"
    "{save_frames    |      | If the input is a camera or a video, save the valid captures to files with pathnames <value>_xxx.png where <value> is the value of this option.}"
    "{size           |<none>| square size.}"
    "{rows           |<none>| number of board's rows.}"
    "{cols           |<none>| number of board's cols.}"
    "{@output        |<none>| filename to save the calculated intrinsics parameters.}"
    "{@input_1       |<none>| first image file, video file or camera index.}"
    "{@input_2       |      | second image file.}"
    "{@input_n       |      | ... n-idx image file.}";

const int ESC_KEY = 27;
const int CAPTURE_KEY = 13;
const int CONTINUE_KEY = 32;

int main(int argc, char *const *argv)
{
    int retCode = EXIT_SUCCESS;

    try
    {
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Calibrate the intrinsic parameters of a camera.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }
        float square_size = parser.get<float>("size");
        int rows = parser.get<int>("rows");
        int cols = parser.get<int>("cols");
        bool is_camera = parser.has("c");
        bool is_video = parser.has("v");
        bool are_images = !is_camera && !is_video;
        bool verbose = parser.has("verbose");
        std::string output_fname = parser.get<cv::String>("@output");
        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }

        // Get the inputs.
        // find the second argument without '-' at begin.
        int input = 1;
        bool found = false;
        while (input < argc && !found)
            found = argv[input++][0] != '-';
        CV_Assert(input < argc);
        cv::VideoCapture video;
        std::vector<std::string> input_fnames;
        std::string save_frames = parser.get<cv::String>("save_frames");
        if (is_camera)
        {
            int camera_idx = 0;
            try
            {
                camera_idx = std::stoi(argv[input]);
            }
            catch (std::exception &e)
            {
                std::cerr << "Error: could not convert the camera index ["
                          << argv[input] << "] to an integer." << std::endl;
                return EXIT_FAILURE;
            }
            video.open(camera_idx);
            if (!video.isOpened())
            {
                std::cerr << "Error: could not open camera with index ["
                          << camera_idx << "]." << std::endl;
                return EXIT_FAILURE;
            }
            video.set(cv::CAP_PROP_FRAME_WIDTH, parser.get<int>("c_width"));
            video.set(cv::CAP_PROP_FRAME_HEIGHT, parser.get<int>("c_height"));
        }
        else if (is_video)
        {
            video.open(argv[input]);
            if (!video.isOpened())
            {
                std::cerr << "Error: could not open video file ["
                          << argv[input - 1] << "]." << std::endl;
                return EXIT_FAILURE;
            }
        }
        else
        {
            for (; input < argc; ++input)
                input_fnames.push_back(std::string(argv[input]));
        }
        cv::Size board_size = cv::Size(cols - 1, rows - 1);
        std::vector<cv::Mat> valid_board_views;
        std::vector<cv::Point3f> _3d_board_points =
            fsiv_generate_3d_calibration_points(board_size, square_size);
        std::vector<std::vector<cv::Point2f>> _2d_points;
        std::vector<std::vector<cv::Point3f>> _3d_points;
        cv::Size camera_size = cv::Size(0, 0);

        const char *wname = nullptr;
        if (verbose || is_camera || is_video)
        {
            wname = "CHESSBOARD";
            cv::namedWindow(wname, cv::WINDOW_GUI_EXPANDED + cv::WINDOW_AUTOSIZE);
        }
        int key = 0;
        for (size_t i = 0; key != ESC_KEY && (!are_images || (i < input_fnames.size())); ++i)
        {
            cv::Mat img;
            if (is_camera)
            {
                do
                {
                    video >> img;
                    if (img.empty())
                    {
                        std::cerr << "Error: could not read frames from camera." << std::endl;
                        return EXIT_FAILURE;
                    }
                    cv::imshow(wname, img);
                    key = cv::waitKey(20) & 0xFF;
                } while (key != ESC_KEY && key != CAPTURE_KEY);
            }
            else if (is_video)
            {
                do
                {
                    video >> img;
                    if (img.empty())
                    {
                        std::cerr << "Error: could not read more frames from video file." << std::endl;
                        std::cerr << "Reset to the start of the video file." << std::endl;
                        video.set(cv::CAP_PROP_POS_FRAMES, 0);
                        continue;
                    }
                    cv::imshow(wname, img);
                    key = cv::waitKey(0) & 0xFF;
                } while (key != ESC_KEY && key != CAPTURE_KEY);
            }
            else
            {
                img = cv::imread(input_fnames[i], cv::IMREAD_COLOR);
                if (img.empty())
                {
                    std::cerr << "Error: could not open image ["
                              << input_fnames[i] << "]." << std::endl;
                    return EXIT_FAILURE;
                }
                if (verbose)
                    cv::imshow(wname, img);
                key = cv::waitKey(20) & 0xFF;
            }

            if (camera_size.area() == 0)
                camera_size = cv::Size(img.cols, img.rows);
            else
            {
                cv::Size img_size = cv::Size(img.cols, img.rows);
                if (img_size != camera_size)
                {
                    std::cerr << "Error: not all the images have the same size."
                              << std::endl;
                    return EXIT_FAILURE;
                }
            }

            std::vector<cv::Point2f> corner_points = std::vector<cv::Point2f>();
            bool was_ok = fsiv_find_chessboard_corners(img, board_size,
                                                       corner_points, wname);
            if (was_ok)
            {
                valid_board_views.push_back(img);
                _3d_points.push_back(_3d_board_points);
                _2d_points.push_back(corner_points);
                if (verbose || is_camera || is_video)
                {
                    std::cout << "Taken the valid view " << std::setfill('0')
                              << std::setw(3) << valid_board_views.size()
                              << std::endl;
                }
            }
        }

        if (verbose | is_camera | is_video)
            cv::destroyWindow(wname);

        if (valid_board_views.size() >= 2)
        {
            cv::Mat camera_matrix;
            cv::Mat dist_coeffs;
            std::vector<cv::Mat> rvects;
            std::vector<cv::Mat> tvects;
            float error = fsiv_calibrate_camera(_2d_points, _3d_points,
                                                camera_size,
                                                camera_matrix, dist_coeffs,
                                                &rvects, &tvects);

            cv::FileStorage fs;
            fs.open(output_fname, cv::FileStorage::WRITE);
            if (!fs.isOpened())
            {
                std::cerr << "Error: could not open [" << output_fname
                          << "] to write." << std::endl;
                return EXIT_FAILURE;
            }
            else
            {
                fsiv_save_calibration_parameters(fs, camera_size, error,
                                                 camera_matrix, dist_coeffs);
                fs.release();
            }

            if (save_frames != "")
            {
                std::ostringstream out;
                const int n_digits = valid_board_views.size() / 10 + 1;
                for (size_t v = 0; v < valid_board_views.size(); ++v)
                {
                    out.str("");
                    out << save_frames << "_" << std::setfill('0') << std::setw(n_digits) << v << ".png";
                    if (!cv::imwrite(out.str(), valid_board_views[v]))
                    {
                        std::cerr << "Error: could not save the view to the filename ["
                                  << out.str() << "]." << std::endl;
                    }
                }
            }

            if (verbose)
            {
                std::ostringstream out;
                const int n_digits = valid_board_views.size() / 10 + 1;
                key = 0;
                for (size_t v = 0; key != ESC_KEY && v < valid_board_views.size(); ++v)
                {
                    out.str("");
                    out << "View " << std::setfill('0') << std::setw(n_digits) << v;
                    cv::drawFrameAxes(valid_board_views[v], camera_matrix, dist_coeffs,
                                      rvects[v], tvects[v], square_size);
                    cv::namedWindow(out.str(), cv::WINDOW_GUI_EXPANDED);
                    cv::imshow(out.str(), valid_board_views[v]);
                    key = cv::waitKey(0) & 0xFF;
                    cv::destroyWindow(out.str());
                }
            }
        }
        else
        {
            std::cerr << "Error: could not find at least two valid views!."
                      << std::endl;
            return EXIT_FAILURE;
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}

/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <exception>

// Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/calib3d/calib3d.hpp>

#include "common_code.hpp"

const char *keys =
    "{help h usage ? |      | print this message.}"
    "{i interactive  |      | Activate interactive mode.}"
    "{f filter       |0     | Laplacian filter to be used to build the sharpening filter: 0->LAP_4, 1->LAP_8, 2->DOG}"
    "{r1             |1     | r1 for DoG filter.}"
    "{r2             |2     | r2 for DoG filter. (0<r1<r2)}"
    "{c circular     |      | use circular convolution.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}";

struct UserData
{
    cv::Mat input;
    std::vector<cv::Mat> input_channels;
    std::vector<cv::Mat> output_channels;
    cv::Mat output;
    int filter_type;
    int r1;
    int r2;
    int only_luma;
    int circular;
    bool interactive;
};

void do_the_work(UserData *user_data)
{
    cv::Mat input = user_data->input;
    if (user_data->input_channels.size() == 3)
        input = user_data->input_channels[2];

    user_data->output = fsiv_image_sharpening(input,
                                              user_data->filter_type,
                                              user_data->r1,
                                              user_data->r2,
                                              user_data->circular);

    if (user_data->input_channels.size() == 3)
    {
        // Revert to BGR.
        cv::Mat hsv;
        user_data->output_channels[2] = user_data->output;
        cv::merge(user_data->output_channels, hsv);
        cv::cvtColor(hsv, user_data->output, cv::COLOR_HSV2BGR);
    }
    if (user_data->interactive)
        cv::imshow("OUTPUT", user_data->output);
}

void filter_trackbar(int pos, void *userdata)
{
    UserData *d = static_cast<UserData *>(userdata);
    pos = std::max(0, std::min(pos, 2));
    d->filter_type = pos;
    std::cout << "Set filter type to " << d->filter_type << std::endl;
    do_the_work(d);
}

void r1_trackbar(int pos, void *userdata)
{
    UserData *d = static_cast<UserData *>(userdata);
    int r1 = std::max(1, std::min(pos, d->r2 - 1));
    int r2 = std::max(r1 + 1, std::min(d->r2, std::min(d->input.rows, d->input.cols) / 2));
    d->r1 = r1;
    d->r2 = r2;
    std::cout << "Set r1=" << d->r1 << " r2=" << d->r2 << std::endl;
    do_the_work(d);
}

void r2_trackbar(int pos, void *userdata)
{
    UserData *d = static_cast<UserData *>(userdata);
    d->r2 = std::max(d->r1 + 1, std::min(pos, std::min(d->input.rows, d->input.cols) / 2));
    std::cout << "Set r1=" << d->r1 << " r2=" << d->r2 << std::endl;
    do_the_work(d);
}

void circular_trackbar(int pos, void *userdata)
{
    UserData *d = static_cast<UserData *>(userdata);
    d->circular = (pos == 1);
    std::cout << "Set circular convolution mode to state " << d->circular
              << std::endl;
    do_the_work(d);
}

int main(int argc, char *const *argv)
{
    int retCode = EXIT_SUCCESS;

    try
    {

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Enhance an image using a sharpening filter. (ver 0.0.0)");
        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }
        UserData data;

        data.filter_type = parser.get<float>("f");
        data.r1 = parser.get<int>("r1");
        data.r2 = parser.get<int>("r2");
        data.interactive = parser.has("i");
        data.circular = parser.has("circular");

        cv::String input_name = parser.get<cv::String>(0);
        cv::String output_name = parser.get<cv::String>(1);

        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }

        data.input = cv::imread(input_name, cv::IMREAD_ANYCOLOR);

        if (data.input.empty())
        {
            std::cerr << "Error: could not open the input image '" << input_name << "'." << std::endl;
            return EXIT_FAILURE;
        }

        data.input.convertTo(data.input, CV_32F, 1.0 / 255.0);

        if (data.input.channels() == 3)
        {
            cv::Mat hsv_img;
            cv::cvtColor(data.input, hsv_img, cv::COLOR_BGR2HSV);
            cv::split(hsv_img, data.input_channels);
            data.output_channels = data.input_channels;
        }

        if (data.filter_type < 0 || data.filter_type > 2)
        {
            std::cerr << "Error: filter type parameter has values in {0, 1, 2}." << std::endl;
            return EXIT_FAILURE;
        }

        if (data.r1 <= 0 || data.r1 >= data.r2)
        {
            std::cerr << "Error: Condition 0 < r1 < r2 is not meet." << std::endl;
            return EXIT_FAILURE;
        }
        int key = 0;

        if (data.interactive)
        {
            cv::namedWindow("INPUT");
            cv::namedWindow("OUTPUT");
            cv::imshow("INPUT", data.input);
            cv::createTrackbar("FILTER", "OUTPUT", 0, 2, filter_trackbar, &data);
            cv::setTrackbarPos("FILTER", "OUTPUT", data.filter_type);
            cv::createTrackbar("R1", "OUTPUT", 0, std::min(data.input.rows, data.input.cols) / 2, r1_trackbar, &data);
            cv::setTrackbarPos("R1", "OUTPUT", data.r2);
            cv::createTrackbar("R2", "OUTPUT", 0, std::min(data.input.rows, data.input.cols) / 2, r2_trackbar, &data);
            cv::setTrackbarPos("R2", "OUTPUT", data.r2);
            cv::createTrackbar("CIRC", "OUTPUT", 0, 1, circular_trackbar, &data);
            cv::setTrackbarPos("CIRC", "OUTPUT", data.circular ? 1 : 0);
            do_the_work(&data);
            key = cv::waitKey(0) & 0xff;
        }
        else
            do_the_work(&data);

        if (key != 27)
        {
            data.output.convertTo(data.output, CV_32F, 256.0);
            if (!cv::imwrite(output_name, data.output))
            {
                std::cerr << "Error: could not save the result in file '" << output_name << "'." << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}

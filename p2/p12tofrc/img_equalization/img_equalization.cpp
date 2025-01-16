#include <iostream>
#include <exception>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "clahe.hpp"

const cv::String keys =
    "{help h usage ? |      | Print this message.}"
    "{i interactive  |      | Activate interactive mode.}"
    "{r radius       |5     | Set the roi size to (2*2^r+1). A value r=0 means global processing.}"
    "{s slope_factor |3.0   | Set the slope factor to control the contrast limitation. A value <1.0 do not do such control.}"
    "{@input         |<none>| Input image.}"
    "{@output        |<none>| Output image.}";

typedef struct
{
  cv::Mat in;
  cv::Mat out;
  bool interactive;
  int r;
  float s;
} UserData;

void do_the_work(UserData *data)
{
  cv::Mat in = data->in;
  std::vector<cv::Mat> channels;
  if (data->in.channels() == 3)
  {
    cv::Mat hsv;
    cv::cvtColor(data->in, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, channels);
    in = channels[2];
  }

  cv::Mat out = fsiv_clahe(in, data->s, data->r);

  if (data->in.channels() == 3)
  {
    cv::Mat hsv;
    channels[2] = out;
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, data->out, cv::COLOR_HSV2BGR);
  }
  else
    data->out = out;

  if (data->interactive)
    cv::imshow("OUTPUT", data->out);
}

void on_change_s(int v, void *data_)
{
  UserData *data = static_cast<UserData *>(data_);
  data->s = v / 10.0;
  std::cerr << "Setting s to " << data->s << std::endl;
  do_the_work(data);
}

void on_change_r(int v, void *data_)
{
  UserData *data = static_cast<UserData *>(data_);
  data->r = v > 0 ? 1 << v : 0;
  std::cerr << "Setting r to " << data->r << std::endl;
  do_the_work(data);
}

int main(int argc, char *const *argv)
{
  int retCode = EXIT_SUCCESS;

  try
  {

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Apply an contrast limited image equalization to the image. (ver 2.0.0)");
    if (parser.has("help"))
    {
      parser.printMessage();
      return 0;
    }

    cv::String input_name = parser.get<cv::String>(0);
    cv::String output_name = parser.get<cv::String>(1);
    int radius = parser.get<int>("r");
    float slope_factor = parser.get<float>("s");
    bool interactive = parser.has("i");

    if (!parser.check())
    {
      parser.printErrors();
      return 0;
    }

    UserData data;
    data.in = cv::imread(input_name, cv::IMREAD_ANYCOLOR);
    if (data.in.empty())
    {
      std::cerr << "Error: could not open the input image." << std::endl;
      exit(-1);
    }

    data.out = data.in.clone();
    data.interactive = interactive;
    data.s = std::max(0.0f, std::min(10.0f, slope_factor));
    radius = std::max(0, std::min(radius, int(std::log(std::min(data.in.rows, data.in.cols)))));
    data.r = radius == 0 ? 0 : 1 << radius;

    int key = 0;

    if (data.interactive)
    {
      cv::namedWindow("INPUT");
      cv::imshow("INPUT", data.in);

      cv::namedWindow("OUTPUT");
      cv::createTrackbar("S", "OUTPUT", 0, 100, on_change_s, &data);
      cv::setTrackbarPos("S", "OUTPUT", std::min(data.s, 10.0f) * 10.0);
      cv::createTrackbar("R", "OUTPUT", 0, int(std::log(std::min(data.in.rows, data.in.cols))), on_change_r, &data);
      cv::setTrackbarPos("R", "OUTPUT", radius);

      key = cv::waitKey(0) & 0xff;
    }
    else
      do_the_work(&data);

    if (key != 27)
    {
      if (!cv::imwrite(output_name, data.out))
      {
        std::cerr << "Error: could not save the result in file '"
                  << output_name << "'." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  catch (...)
  {
    std::cerr << "Capturada excepcion desconocida!" << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}

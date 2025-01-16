/*!
 This program shows the images of the dataset.

 2024: UPTATED by Rafael Berral

*/

#include <iostream>
#include <string>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset.hpp"

const char * keys =
    "{help h usage ? |      | print this message   }"
    "{@data          |<none>| folder with the dataset.}"
    ;

// Check if this is a Windows system
#ifdef _WIN32
   // Arrow keys are different in Windows: ASCII codes of 'n' and 'm'
    static const int LEFT_ARROW = 110;
    static const int RIGHT_ARROW = 109;
#else
    static const int LEFT_ARROW = 81;
    static const int RIGHT_ARROW = 83;
#endif

int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;

  try {
      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Load the BAA500 dataset. ");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }
      std::string folder = parser.get<std::string>("@data");
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }
      
      std::cout << "Loading data from folder: " << folder << std::endl;
      
      cv::Mat X, y;
      fsiv_load_dataset(folder, X, y);
      const int dataset_size = X.rows;
      std::cout << "Loaded " << dataset_size << " images." << std::endl;

      // TODO: print the dimensionalities of X and y
      // ...
      
      int key = 0;
      int idx = 0;
      std::string wname = "IMAGE";
      cv::namedWindow(wname, cv::WINDOW_GUI_EXPANDED);
      cv::resizeWindow(wname, cv::Size(256,256));
      do
      {          
          cv::Mat img_v = X.row(idx);
          // TODO: use reshape to convert the row into a NxN image
          cv::Mat img;
          // ...
                    
          cv::imshow(wname, img);

          // TODO: Print the label of the current image: fsiv_get_dataset_label_name()
          // ...
          // std::cout << "Idx " << idx << ": "                    
             
          key = cv::waitKey(0) & 0xff;
          
          if (key == LEFT_ARROW)
              idx = (idx-1+dataset_size) % dataset_size;
          else if  (key == RIGHT_ARROW)
              idx = (idx+1) % dataset_size;
          else if (key != 27)
              std::cout << "Unknown keypress code '" << key
                        << "' [Press <-, ->, or ESC]." << std::endl;
      }
      while (key != 27);

      cv::destroyWindow(wname);
  }
  catch (std::exception& e)
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

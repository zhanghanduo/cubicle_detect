/*
 * yolo_obstacle_detector_node.cpp
 *
 *  Created on: June 19, 2018
 *      Author: Zhang Handuo
 *   Institute: NTU, ST Corp Lab
 */


#include <darknet_ros/YoloObjectDetector.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

//#include <message_filters/sync_policies/exact_time.h>

using namespace message_filters;

void readFoldernames(std::vector<cv::String> &foldernames, const cv::String &directory)
{
  DIR *dir;
  class dirent *ent;
  class stat st;

  dir = opendir(directory.c_str());
  while ((ent = readdir(dir)) != NULL) {
    const std::string file_name = ent->d_name;
    const std::string full_file_name = directory + "/" + file_name ;

    if (file_name[0] == '.')
      continue;

    if (stat(full_file_name.c_str(), &st) == -1)
      continue;

    const bool is_directory = (st.st_mode & S_IFDIR) != 0;

    if (is_directory) {
      foldernames.push_back(file_name);
    }

//    size_t pos = file_name.find(extension);
//
//    if (pos!=std::string::npos) {
//      std::string file_name_no_ext = file_name.substr (0,pos);
//      filenames.push_back(file_name_no_ext);
//    }

  }
  closedir(dir);
}

void readFilenames(std::vector<cv::String> &filenames, const cv::String &directory, const cv::String &extension)
{
  DIR *dir;
  class dirent *ent;
  class stat st;

  dir = opendir(directory.c_str());
  while ((ent = readdir(dir)) != NULL) {
    const std::string file_name = ent->d_name;
    const std::string full_file_name = directory + "/" + file_name ;

    if (file_name[0] == '.')
      continue;

    if (stat(full_file_name.c_str(), &st) == -1)
      continue;

    const bool is_directory = (st.st_mode & S_IFDIR) != 0;

    if (is_directory)
      continue;

    size_t pos = file_name.find(extension);

    if (pos!=std::string::npos) {
      std::string file_name_no_ext = file_name.substr (0,pos);
      filenames.push_back(file_name_no_ext);
    }

  }
  closedir(dir);
}

bool compare_filenames(std::string a, std::string b)
{
  if(a.length()==b.length())
    return a < b;
  else
    return a.length()<b.length();
}

int main(int argc, char** argv) {
//  std::cout<<"Debug main() 1.0"<<std::endl;
  ros::init(argc, argv, "cubicle_detect");
  ros::NodeHandle nh_pub;
  ros::NodeHandle nodeHandle("~");

  std::string image_left_topic, image_right_topic, image_left_info, image_right_info, mainFolder;

  image_left_topic = "/long/left/image_rect";

  image_right_topic = "/long/right/image_rect";

  image_left_info = "/long/left/camera_info";

  image_right_info = "/long/right/camera_info";

  mainFolder = "/home/ugv/catkin_ws/src/cubicle_detect/darknet_ros/data";

//  std::cout<<"Debug main() before YoloObjectDetector initiation"<<std::endl;
  darknet_ros::YoloObjectDetector detector(nodeHandle, nh_pub);
//  std::cout<<"Debug main() after YoloObjectDetector initiation"<<std::endl;

  if(nodeHandle.getParam("image_left_topic", image_left_topic))
    ROS_INFO("Get left image topic: %s", image_left_topic.c_str());

  if(nodeHandle.getParam("image_right_topic", image_right_topic))
    ROS_INFO("Get right image topic: %s", image_right_topic.c_str());

  if(nodeHandle.getParam("image_info_left", image_left_info))
    ROS_INFO("Get left image topic: %s", image_left_info.c_str());

  if(nodeHandle.getParam("image_info_right", image_right_info))
    ROS_INFO("Get right image info topic: %s", image_right_info.c_str());

  if(nodeHandle.getParam("data_path", mainFolder))
    ROS_INFO("Get data_path info topic: %s", mainFolder.c_str());



//  sensor_msgs::CameraInfoConstPtr left_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_left_info);
//  sensor_msgs::CameraInfoConstPtr right_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_right_info);

//  std::cout<<"Debug main() before loadCameraCalibration"<<std::endl;

//  detector.loadCameraCalibration(left_info, right_info);
//  detector.init();

//  std::cout<<"Debug main() before DefineLUTs"<<std::endl;
//  detector.DefineLUTs();

//  std::cout<<"Debug main() after DefineLUTs"<<std::endl;

//  Subscriber<sensor_msgs::Image> image1_sub(nh_pub, image_left_topic, 20);
//  Subscriber<sensor_msgs::Image> image2_sub(nh_pub, image_right_topic, 20);
//  Subscriber<sensor_msgs::Image> image3_sub(nh_pub, "/kitti/camera_color_left/image_raw", 20);
//  Subscriber<sensor_msgs::CameraInfo> sub_info_l_(nh_pub, image_left_info, 20);
//  Subscriber<sensor_msgs::CameraInfo> sub_info_r_(nh_pub, image_right_info, 20);

//  typedef sync_policies::ApproximateTime<sensor_msgs::Image,
//          sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
//  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10),
//                                  image1_sub, image2_sub, image3_sub);

//  sync.registerCallback(boost::bind(&darknet_ros::YoloObjectDetector::cameraCallback,
//                                    &detector ,_1, _2, _3));

//  std::cout<<"Debug main() after camerCallBack"<<std::endl;

//  ros::spin();

  std::vector<cv::String> foldernames, filenames;
  cv::String ext = ".jpg";
  cv::String file_ext = "/det/det.txt";
  readFoldernames(foldernames, mainFolder);
  for(int ii=0;ii<foldernames.size();ii++){
    const std::string folderName = mainFolder +"/" + foldernames[ii] +"/img1/";
    std::cout<<foldernames[ii]<<std::endl;
    readFilenames(filenames, folderName, ext);
    sort(filenames.begin(), filenames.end(), compare_filenames);
    detector.setFileName(foldernames[ii]);

    const std::string detFileName = mainFolder +"/" + foldernames[ii] + file_ext;
    detector.readDetections(detFileName);

    for(size_t frame = 0; frame < filenames.size(); ++frame) {
      std::string fName = filenames[frame];
//      std::cout <<std::endl<< fName << std::endl<<std::endl;
      cv::Mat image = imread(folderName + fName + ext);
      if (!image.data)
        std::cerr << "Problem loading image!!!" << std::endl;
      else
        detector.readImage(image,fName);
    }
    detector.closeFile();
//    std::cout<<std::endl;
    filenames.clear();
  }

  detector.getFrameRate();

  return 0;
}

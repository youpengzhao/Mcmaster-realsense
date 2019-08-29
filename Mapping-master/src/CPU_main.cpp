// g++ -std=c++11 CPU_main.cpp -o CPU_main -lrealsense2 -lboost_iostreams -lboost_system -lboost_filesystem `pkg-config opencv --cflags --libs` -lpthread
//g++ -std=c++11 CPU_main.cpp -o Cpu_main  -lboost_iostreams -lboost_system -lboost_filesystem `pkg-config --libs opencv4` `pkg-config --cflags opencv4` `pkg-config --libs realsense2` `pkg-config --cflags realsense2` -lpthread

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <time.h>

#include "../include/Voxel.hpp"
#include "../include/Logging.hpp"



int main(int argc, char const *argv[])
{
    std::atomic_bool alive {true};

    /* Map Front End */
    Map_FE * F = new CPU_FE();

    /* Camera Initialization */
    Camera C;
    Bool_Init bC = C.Init();
    if (bC.t265 && bC.d435)
        std::cout << "Cameras initialized\n";
    else 
        std::cout << "Atleast one camera is not connected\n";

    /* Logger Initialization */
    Logger L;
    L.Init();
    
    rs2::threshold_filter thr_filter;
    rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
    rs2::temporal_filter temp_filter;      //Temporal filter isn't very usful.If needed, some settings should be changed.

    thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.11);       //We only use spatial filter and threshold filter.
    thr_filter.set_option( RS2_OPTION_MAX_DISTANCE, 2.0);      //the max and min distance is related to D435_MIN and D435_MAX in camera.hpp
    spat_filter.set_option( RS2_OPTION_FILTER_MAGNITUDE,4);
    spat_filter.set_option( RS2_OPTION_FILTER_SMOOTH_ALPHA,0.7);
    spat_filter.set_option( RS2_OPTION_FILTER_SMOOTH_DELTA,18);
    //temp_filter.set_option( RS2_OPTION_FILTER_SMOOTH_DELTA,50);     
    //temp_filter.set_option( RS2_OPTION_FILTER_SMOOTH_ALPHA,0.5);
    
   /* Thread for checking exit condition */

    std::thread exit_check([&]() {
        while (alive) {
            if (std::cin.get() == ' ') {
                cv::destroyAllWindows();
                alive = false;
            }
        }
    });

    /* Thread for receiving frames and storing them as video and csv files */ 

    std::thread rxFrame([&]() {
        while (alive) {
            auto sleep_start = std::chrono::high_resolution_clock::now();

			auto tframe = C.pipelines[0].wait_for_frames();       //tframe's and dframe's type is frameset. 
			auto dframe = C.pipelines[1].wait_for_frames();

			auto t = tframe.first_or_default(RS2_STREAM_POSE);          //Get the first frame of the frameset.
			auto d = dframe.get_depth_frame();

			if (!t||!d) { std::cout<<2<<std::endl; continue;}
			C.t_queue.enqueue(tframe);                  //Enqueue the frameset.
			C.d_queue.enqueue(dframe);

            // sleep for remaining time
            auto time_sleep = std::chrono::high_resolution_clock::now() - sleep_start;
            double time_s = std::chrono::duration_cast<std::chrono::milliseconds>(time_sleep).count();
            if ((1000.0/INPUT_RATE)-time_s > 0){
                usleep((1000.0/INPUT_RATE-time_s) * 1000);
            }
            // std::cout << time_s << "\n";
		}
    });


    rs2::frameset t_frameset,d_frameset;
    auto start = std::chrono::high_resolution_clock::now();

    while (alive) {
    	C.t_queue.poll_for_frame(&t_frameset);       //Poll for the new frame and dequeue it if there is.
    	C.d_queue.poll_for_frame(&d_frameset);

    	if (t_frameset && d_frameset) {
    		auto depthFrame = d_frameset.get_depth_frame();
                // Note the concatenation of output/input frame to build up a chain
     		rs2::frame filtered = depthFrame;
       		filtered = thr_filter.process(filtered);
      		filtered = spat_filter.process(filtered);
       		//filtered = temp_filter.process(filtered);
    		auto poseFrame  = t_frameset.first_or_default(RS2_STREAM_POSE);

    		cv::Mat depth(cv::Size(w, h), CV_16UC1, (void *)filtered.get_data(), cv::Mat::AUTO_STEP);   //Tramsform the frame into cv::Mat
    		auto pose = poseFrame.as<rs2::pose_frame>().get_pose_data();   //Get pose information from T265 in the type of rs2_pose.
                
            /* update global map */
            F->Update (C, pose, depth);
            /*                   */

    	    auto elapsed = std::chrono::high_resolution_clock::now() - start;
            float milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            //std::cout << milliseconds << "\n";

            L.Log(&C, &pose, &depth);

    	}

    	start = std::chrono::high_resolution_clock::now();

    }

    rxFrame.join();

    L.Close(&C, F);

    std::cout << "Program terminated sucessfully\n";
	return 0;
	
}

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>
#include <dirent.h> //mkdir
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include <ros/ros.h>
#include <upo_fcn_learning/capture.h>

//TF
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

//open a directory
#include <boost/filesystem.hpp>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <laser_geometry/laser_geometry.h>
#include <nav_msgs/GetMap.h>
//PCL
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include "pcl_ros/transforms.h"
#include <pcl/register_point_struct.h>

//people message
#include <upo_msgs/PersonPoseArrayUPO.h>
#include <upo_msgs/PersonPoseUPO.h>

//rosbag 
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


using namespace std;


tf::TransformListener* 		tf_;
string 						bag_dir_;
Capture*					capture_;






float normalizeAngle(float val, float min, float max) {
	
	float norm = 0.0;
	if (val >= min)
		norm = min + fmod((val - min), (max-min));
	else
		norm = max - fmod((min - val), (max-min));
            
    return norm;
}



void transformToLocal(vector<geometry_msgs::PoseStamped>* agent, vector<upo_msgs::PersonPoseArrayUPO>* people, vector<geometry_msgs::PoseStamped>* goal, vector<sensor_msgs::PointCloud>* obstacles)
{

	//All the input data has to be in the same coordinate frame
	string local_frame = "base_link";
	string data_frame = agent->at(0).header.frame_id;
	if(data_frame == local_frame || data_frame == ("/"+local_frame)) {
		cout << "TransformToLocal. Coordinates of data already in the local frame (base_link)" << endl;
		return; 
	}

	geometry_msgs::PoseStamped origin = agent->at(0);
	for(unsigned int i=0; i<agent->size(); i++)
	{
		//take the robot position
		geometry_msgs::PoseStamped pose = agent->at(i);
		float oth = tf::getYaw(pose.pose.orientation);
		float ox = pose.pose.position.x;
		float oy = pose.pose.position.y;

		
		//-----transform the people to robot frame----
		if(people->size() > i) {
			people->at(i).header.frame_id = local_frame;
			vector<upo_msgs::PersonPoseUPO> p = people->at(i).personPoses;
			for(unsigned int j=0; j<p.size(); j++)
			{
				p[j].header.frame_id = local_frame;
				float pth = tf::getYaw(p[j].orientation);
				float px = p[j].position.x;
				float py = p[j].position.y;
				/*
				Transform the person location into robot location frame: 
											|cos(th)  sin(th)  0|
					Rotation matrix R(th)= 	|-sin(th) cos(th)  0|
											|  0        0      1|
									             
					x' = (xr-xp)*cos(th_p)+(yr-yp)*sin(th_p)
					y' = (xr-xp)*(-sin(th_p))+(yr-yp)*cos(th_p)
				*/
				float nx = (px - ox)*cos(oth) + (py - oy)*sin(oth);
				float ny = (px - ox)*(-sin(oth)) + (py - oy)*cos(oth);			
				float nth = normalizeAngle((pth - oth), -M_PI, M_PI);

				people->at(i).personPoses[j].position.x = nx;
				people->at(i).personPoses[j].position.y = ny;
				people->at(i).personPoses[j].orientation = tf::createQuaternionMsgFromYaw(nth);

			}
		}

		//-----transform goal to robot frame-----
		if(goal->size() > i) {
			goal->at(i).header.frame_id = local_frame;
			float gth = tf::getYaw(goal->at(i).pose.orientation);
			float gx = goal->at(i).pose.position.x;
			float gy = goal->at(i).pose.position.y;

			float nx = (gx - ox)*cos(oth) + (gy - oy)*sin(oth);
			float ny = (gx - ox)*(-sin(oth)) + (gy - oy)*cos(oth);			
			float nth = normalizeAngle((gth - oth), -M_PI, M_PI);
			goal->at(i).pose.position.x = nx;
			goal->at(i).pose.position.y = ny;
			goal->at(i).pose.orientation = tf::createQuaternionMsgFromYaw(nth);
		}


		//-----Transform obstacles to robot frame----
		if(obstacles->size() > i) {
			obstacles->at(i).header.frame_id = local_frame;
			for(unsigned int j=0; j<obstacles->at(i).points.size(); j++) {
				float bx = obstacles->at(i).points[j].x;
				float by = obstacles->at(i).points[j].y;
				float nx = (bx - ox)*cos(oth) + (by - oy)*sin(oth);
				float ny = (bx - ox)*(-sin(oth)) + (by - oy)*cos(oth);
				obstacles->at(i).points[j].x = nx;
				obstacles->at(i).points[j].y = ny;
			}
		}

	
		//-----transform path (robot) to robot frame----
		//First point in the path is the origin (0,0,0)
		if(i==0) {
			geometry_msgs::PoseStamped ini;
			ini.header.stamp = origin.header.stamp;
			ini.header.frame_id = local_frame;
			ini.pose.position.x = 0.0;
			ini.pose.position.y = 0.0;
			ini.pose.position.z = 0.0;
			ini.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);
			agent->at(i) = ini;
		}else {
			agent->at(i).header.frame_id = local_frame;
			float ax = agent->at(i).pose.position.x;
			float ay = agent->at(i).pose.position.y;
			float ath = tf::getYaw(agent->at(i).pose.orientation);
			float orith = tf::getYaw(origin.pose.orientation);
			float nx = (ax - origin.pose.position.x)*cos(orith) + (ay - origin.pose.position.y)*sin(orith);
			float ny = (ax - origin.pose.position.x)*(-sin(orith)) + (ay - origin.pose.position.x)*cos(orith);
			float nth = normalizeAngle((ath - orith), -M_PI, M_PI);
			agent->at(i).pose.position.x = nx;
			agent->at(i).pose.position.y = ny;
			agent->at(i).pose.orientation = tf::createQuaternionMsgFromYaw(nth);
		}
	}

}






/**
 * Open the directory with the demonstration trajectories recorded in bag files.
 * IMPORTANT: the coordinate frame must be the same for all the data.
 */
bool openBags(std::string dir)
{
	// we check if the directory is valid
	boost::filesystem::path my_path(dir.c_str());
	if(!boost::filesystem::exists(my_path) || !boost::filesystem::is_directory(my_path))
	{
		ROS_ERROR("ERROR. Directory '%s' of demonstration samples does not exists or is not a directory", dir.c_str());
		return false;
	}else if(boost::filesystem::is_empty(my_path)) {
		ROS_ERROR("ERROR. Directory of demonstration samples is empty.");
		return false;	
	}
 
	int scenarios = 0;
	boost::filesystem::directory_iterator it_end;
	for(boost::filesystem::directory_iterator it_sc(my_path); it_sc != it_end; it_sc++ )
		scenarios++;

	printf("Bags detected: %u\n", scenarios);	

	int nsc = 0;
	//int num_traj = 1;

	try{
		// Get the files in the main directory
		boost::filesystem::directory_iterator end_it;
		for(boost::filesystem::directory_iterator it_files(my_path); it_files != end_it; it_files++ ) 
		{

			vector<geometry_msgs::PoseStamped> robot;		//robot poses
			vector<geometry_msgs::PoseStamped> goal;		//Goal pose
			vector<upo_msgs::PersonPoseArrayUPO> people;	//People poses
			vector<sensor_msgs::PointCloud> obstacles;		//obstacles poses

			string filename;

			if( boost::filesystem::is_regular_file(it_files->status())) 
			{
				filename = string(it_files->path().filename().c_str());
				printf("\nScenario %u. File name: %s\n", (nsc+1), filename.c_str());
				string bag_file = string(it_files->path().c_str()); 
				rosbag::Bag bag;
				try {
					bag.open(bag_file, rosbag::bagmode::Read);
							
				} catch (rosbag::BagException& ex) {
					ROS_ERROR("Error opening bag file %s : %s", it_files->path().filename().c_str(), ex.what());
					return false;
				}

				vector<string> topics;
				topics.push_back(string("robot"));
				topics.push_back(string("people"));
				topics.push_back(string("goal"));
				topics.push_back(string("obstacles"));
				rosbag::View view(bag, rosbag::TopicQuery(topics));
						
						
				foreach(rosbag::MessageInstance const m, view)
				{
					if(m.getTopic()=="robot"){
						geometry_msgs::PoseStamped::Ptr robo = m.instantiate<geometry_msgs::PoseStamped>();
						robot.push_back(*robo.get());
					}
					else if(m.getTopic()=="people"){
						upo_msgs::PersonPoseArrayUPO::Ptr ppl = m.instantiate<upo_msgs::PersonPoseArrayUPO>();
						people.push_back(*ppl.get());
					}
					else if (m.getTopic()=="goal"){
						geometry_msgs::PoseStamped::Ptr gol = m.instantiate<geometry_msgs::PoseStamped>();
						goal.push_back(*gol.get());
					}
					else if (m.getTopic()=="obstacles"){
						sensor_msgs::PointCloud2::Ptr obs = m.instantiate<sensor_msgs::PointCloud2>();
						//Transform pointCloud2 to pointCloud
						sensor_msgs::PointCloud2 pc2 = *obs.get();
						sensor_msgs::PointCloud temp_pc;
						bool done = sensor_msgs::convertPointCloud2ToPointCloud(pc2, temp_pc);
						if(done)  
							obstacles.push_back(temp_pc);
						else
							ROS_ERROR("\n\nopenBags. ERROR in convertPointCloud2toPoingCloud!!!!!\n");
						
						
					}
				}

				bag.close();
				nsc++;
				

			} else
			{
				ROS_ERROR("File %s, is not a regular file", it_files->path().filename().c_str());
				return false;
			}

			//Transform the data to robot frame (in case they are not yet) 
			transformToLocal(&robot, &people, &goal, &obstacles);

			//Set the scenario, create and store the images
			capture_->generateImg(&obstacles[0], &people[0].personPoses, &goal[0], &robot, filename);

			
		} //end of the for loop demonstration files 
				
				

	} catch(const boost::filesystem::filesystem_error& ex){
		std::cout << "ERROR opening demonstrations files: " << ex.what() << '\n';
	}

	return true;
}





int main(int argc, char** argv) {

	ros::init(argc, argv, "-Rosbag data to images-");
  	
	//tf_listener = new tf::TransformListener(ros::Duration(10));

	capture_ = new Capture();

	ros::NodeHandle n("~");
	ros::NodeHandle nh;
	
	n.param("bag_directory", bag_dir_, string("/home/"));

	openBags(bag_dir_);

}

	


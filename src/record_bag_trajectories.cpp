
//Read/write a file
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <math.h>
#include <random>

//open a directory
#include <boost/filesystem.hpp>

//msg irl weights
//#include <upo_navigation/FeaturesWeights.h>

#include <mutex>  /* Mutex */

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/MapMetaData.h>
#include <nav_msgs/GetMap.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <upo_msgs/PersonPoseUPO.h>
#include <upo_msgs/PersonPoseArrayUPO.h>

//PCL
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include "pcl_ros/transforms.h"
#include <pcl/register_point_struct.h>

#include <visualization_msgs/Marker.h>

//Services
#include <navigation_features/PoseValid.h>
#include <upo_rrt_planners/MakePlan.h>

//rosbag 
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;


tf::TransformListener* 			tf_;

unsigned int 					scenarios_ = 0;
string 							store_frame_;
string							store_dir_;

bool 							use_rrt_path_ = false;
bool 							recording_ = false;
bool							random_scenario_ = true;

//map data
double				 			width_; // m
double							height_; // m
double 							resolution_; // m/cell
std::vector<float> 				origin_;

//upo_nav::UpoNavigation* 		UpoNav_;

//Services 
ros::ServiceClient				valid_client_;
ros::ServiceClient				plan_client_;


//publish the initial robot position
ros::Publisher 					initial_pose_pub_;
ros::Publisher					people_pose_pub_;
ros::Publisher 					goal_marker_pub_;


//subscriptions
ros::Subscriber clicked_point_sub_;		//Goal
ros::Subscriber rrt_goal_sub_;			//Goal2
ros::Subscriber odom_pose_sub_;			//odom (robot pose)
ros::Subscriber people_pose_sub_;		//people
ros::Subscriber laser_sub_;				//laser (obstacles)
ros::Subscriber pc_sub_;				//pointcloud (obstacles)
ros::Subscriber vel_sub_;				//robot vel commands


//people poses
upo_msgs::PersonPoseArrayUPO people_;
mutex peopleMutex_;

//goal pose
geometry_msgs::PoseStamped goal_;
mutex goalMutex_;

//robot pose
geometry_msgs::PoseStamped robot_current_;
geometry_msgs::PoseStamped robot_map_current_;
mutex robotMutex_;

//Cmd vel
geometry_msgs::Twist cmd_vel_;
mutex velMutex_;

//laser
sensor_msgs::LaserScan laser_;
mutex laserMutex_;

//Point cloud
sensor_msgs::PointCloud2 pc_;
mutex pcMutex_;


struct trajectory_t {
	vector<geometry_msgs::PoseStamped> goal_data;
	vector<upo_msgs::PersonPoseArrayUPO> people_data;
	vector<sensor_msgs::PointCloud2> obs_data;
	vector<geometry_msgs::PoseStamped> robot_data;
	vector<geometry_msgs::Twist> vel_data;
	vector<geometry_msgs::PoseStamped> robot_map_data;
};

trajectory_t path_;


//Initialize random numbers generation
std::random_device rd;
std::mt19937 gen(rd());
uniform_real_distribution<double> distribution_real(0.0,1.0);
uniform_int_distribution<int> distribution_int(1,7); //number of people



// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
    //strftime(buf, sizeof(buf), "%X", &tstruct);

    return buf;
}




bool isQuaternionValid(const geometry_msgs::Quaternion q){
    //first we need to check if the quaternion has nan's or infs
    if(!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) || !std::isfinite(q.w)){
		ROS_ERROR("Quaternion has infs!!!!");
		return false;
    }
    if(std::isnan(q.x) || std::isnan(q.y) || std::isnan(q.z) || std::isnan(q.w)) {
		ROS_ERROR("Quaternion has nans !!!");
		return false;
	}
	
	if(fabs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1) > 0.01) {
		ROS_ERROR("Quaternion malformed, magnitude: %.3f should be 1.0", (q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w));
		return false;
	}

    tf::Quaternion tf_q(q.x, q.y, q.z, q.w);

    //next, we need to check if the length of the quaternion is close to zero
    if(tf_q.length2() < 1e-6){
      ROS_ERROR("Quaternion has length close to zero... discarding.");
      return false;
    }

    //next, we'll normalize the quaternion and check that it transforms the vertical vector correctly
    tf_q.normalize();

    tf::Vector3 up(0, 0, 1);

    double dot = up.dot(up.rotate(tf_q.getAxis(), tf_q.getAngle()));

    if(fabs(dot - 1) > 1e-3){
      ROS_ERROR("Quaternion is invalid... for navigation the z-axis of the quaternion must be close to vertical.");
      return false;
    }

    return true;
}




geometry_msgs::PoseStamped transformPoseTo(geometry_msgs::PoseStamped in, string outframe)
{
	//Transform to the requested frame
	geometry_msgs::PoseStamped pose_in = in;
	geometry_msgs::PoseStamped pose_out = in;

	geometry_msgs::Quaternion q = pose_in.pose.orientation;
	if(!isQuaternionValid(q))
	{
		ROS_WARN("record_bag_trajectories. transformPoseTo. Quaternion no valid. Creating new quaternion with yaw=0.0");
		pose_in.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);
	}

	try {
		tf_->transformPose(outframe.c_str(), pose_in, pose_out);
	}catch (tf::TransformException ex){
		ROS_ERROR("record_bag_trajectories. transformPoseTo. Exception: %s",ex.what());
		return pose_out;
	}
	return pose_out;
}




//void amclReceived(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
	ROS_INFO_ONCE("OdomCallback. robot position received!");
	//printf("odom received\n");
	//if(recording_)
	//{
		geometry_msgs::PoseStamped robot_next;
		robot_next.header = msg->header;
		robot_next.header.stamp = ros::Time();
		robot_next.pose = msg->pose.pose;
		robot_next.pose.orientation = tf::createQuaternionMsgFromYaw(tf::getYaw(msg->pose.pose.orientation));
		//Transform to the requested frame
		geometry_msgs::PoseStamped pose_out = transformPoseTo(robot_next, store_frame_);
		geometry_msgs::PoseStamped pose_map_out = transformPoseTo(robot_next, "/map");
		robotMutex_.lock();
		robot_current_ = pose_out;
		robot_map_current_ = pose_map_out;
		robotMutex_.unlock();
	//}
		
}



void clickedPointCallback(const geometry_msgs::PointStamped::ConstPtr& msg)
{
	printf("clickedPoint received\n");
	goalMutex_.lock();
	
	goal_.header.stamp = msg->header.stamp;
	goal_.header.frame_id = msg->header.frame_id; //map
	goal_.pose.position.x = msg->point.x;
	goal_.pose.position.y = msg->point.y;
	goal_.pose.position.z = 0.0;
	goal_.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);

	//Transform goal to the required frame
	geometry_msgs::PoseStamped pose_out = transformPoseTo(goal_, store_frame_);   
	goal_ = pose_out;

	
	//Clear the vectors
	if(path_.robot_data.size() > 0) {
		path_.robot_data.clear();
		path_.people_data.clear();
		path_.goal_data.clear();
		path_.obs_data.clear();
		path_.vel_data.clear();
	}
	
	//Publish a visualization marker with the goal
	visualization_msgs::Marker marker;
	marker.header.frame_id = msg->header.frame_id;
	marker.header.stamp = ros::Time::now();
	marker.ns = "basic_shapes";
	marker.id = 0;
	marker.type = visualization_msgs::Marker::CYLINDER;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x = msg->point.x;
	marker.pose.position.y = msg->point.y;
	marker.pose.position.z = 0.2;
	marker.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);
	marker.scale.x = 0.3;
	marker.scale.y = 0.3;
	marker.scale.z = 0.1;
	marker.color.r = 1.0f;
	marker.color.g = 0.0f;
	marker.color.b = 0.0f;
	marker.color.a = 1.0;
	marker.lifetime = ros::Duration();
	goal_marker_pub_.publish(marker); 
	
	recording_ = true;
	
	goalMutex_.unlock();
	
}





bool randomNewRobotPose()
{
	bool ok = false;
	geometry_msgs::PoseStamped rpose;
	rpose.header.frame_id = "map";
	rpose.header.stamp = ros::Time::now();
	rpose.pose.position.z = 0.0;

	int i=0;
	do {
 		i++;
		double rx = (distribution_real(gen)*width_) - origin_[0];
		double ry = (distribution_real(gen)*height_) - origin_[1];

		//orientation
		double rth = distribution_real(gen)*M_PI;
		if(distribution_real(gen) > 0.5)
			rth = -rth;

		//Check if the random point is valid
		rpose.pose.position.x = rx;
		rpose.pose.position.y = ry;
		rpose.pose.orientation = tf::createQuaternionMsgFromYaw(rth);
		navigation_features::PoseValid pv;
		pv.request.pose = rpose;
		if(!valid_client_.call(pv))
		{
			ROS_ERROR("record_bag_trajectories. Error calling service 'is_pose_valid'");
			return false;
		}
		ok = pv.response.ok;
		//if(!ok)
			//printf("robot new pose x:%.2f, y:%.2f not valid!\n", rx, ry);

	} while(!ok && i<2000);  

	if(!ok)
		return false;	

	//Publish robot position
	geometry_msgs::PoseWithCovarianceStamped new_pose;
	new_pose.header = rpose.header;
	new_pose.pose.pose = rpose.pose;
	new_pose.pose.covariance[0]=0.25;
	new_pose.pose.covariance[7]=0.25;
	new_pose.pose.covariance[35]=0.06853891945200942;
	initial_pose_pub_.publish(new_pose);
	
	printf("New AGENT position generated. frame:%s, x:%.2f, y:%.2f, th:%.2f\n", rpose.header.frame_id.c_str(), rpose.pose.position.x, rpose.pose.position.y, tf::getYaw(rpose.pose.orientation));

	robotMutex_.lock();
	robot_current_ = rpose;
	robotMutex_.unlock();
	return true;
}







bool randomNewPeople()
{
	//Take the current robot position
	robotMutex_.lock();
	geometry_msgs::PoseStamped robot = robot_current_;
	robotMutex_.unlock();
	float x = robot.pose.position.x;
	float y = robot.pose.position.y;
	bool ok = false;

	geometry_msgs::PoseStamped person;
	person.header = robot.header;
	person.pose.position.z = 0.0;

	ros::Time time = ros::Time::now();
	
	upo_msgs::PersonPoseArrayUPO peop;
	int np = distribution_int(gen); //random number of people [0,3]
	peop.header.stamp = time;
	peop.header.frame_id = robot.header.frame_id;
	if(np>4)
		np = 4;

	peop.size = np;
	printf("Generating %i new people...\n", np);
	for(unsigned int j=0; j<np; j++)
	{
		int i=0;
		do {
			i++;
			double rx = distribution_real(gen)*4.4 + 0.5;
			if(distribution_real(gen) > 0.5)
				rx = -rx;
			double ry = distribution_real(gen)*4.4 + 0.5;
			if(distribution_real(gen) > 0.5)
				ry = -ry;
			//orientation
			double rth = distribution_real(gen)*M_PI;
			if(distribution_real(gen) > 0.5)
				rth = -rth;

			//Check if the random point is valid
			person.pose.position.x = x + rx;
			person.pose.position.y = y + ry;
			person.pose.orientation = tf::createQuaternionMsgFromYaw(rth);
			navigation_features::PoseValid pv;
			pv.request.pose = person;
			if(!valid_client_.call(pv))
			{
				ROS_ERROR("record_bag_trajectories. Error calling service 'is_pose_valid'");
				return false;
			}
			ok = pv.response.ok;

		} while(!ok && i<2000);
		
 		if(ok)		 
		{
			//Create the person and store him
	 		upo_msgs::PersonPoseUPO p;
			p.header.frame_id = peop.header.frame_id;
			p.header.stamp = time; 
			p.id = (j+1);
			p.vel = 0.0;
			p.position = person.pose.position;
			p.orientation = person.pose.orientation;

			peop.personPoses.push_back(p);

			printf("New PERSON position generated. frame:%s, x:%.2f, y:%.2f, th:%.2f\n\n", p.header.frame_id.c_str(), p.position.x, p.position.y, tf::getYaw(p.orientation));
		}
	} 

	if(peop.personPoses.empty())
		return false;

	//Publish the people
	people_pose_pub_.publish(peop);

	peopleMutex_.lock();
	people_ = peop;
	peopleMutex_.unlock();

	return true;
}



bool randomNewGoal()
{
	//Take the current robot position
	robotMutex_.lock();
	geometry_msgs::PoseStamped robot = robot_current_;
	robotMutex_.unlock();

	//Calculate a random position around the robot (between 2 and 4.5 meters)
	geometry_msgs::PoseStamped gpose;
	gpose.header = robot.header;
	gpose.pose.position.z = 0.0;
	//gpose.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);
	gpose.pose.orientation.x = 0.0;
	gpose.pose.orientation.y = 0.0;
	gpose.pose.orientation.z = 0.0;
	gpose.pose.orientation.w = 1.0;
	float x = robot.pose.position.x;
	float y = robot.pose.position.y;
	bool ok = false;

	int i=0;
	do {
		i++;
		double rx = distribution_real(gen)*2.5 + 2.0;
		if(distribution_real(gen) > 0.5)
			rx = -rx;
		double ry = distribution_real(gen)*2.5 + 2.0;
		if(distribution_real(gen) > 0.5)
			ry = -ry;

		float xn = x + rx;
		float yn = y + ry;

		//Check if the random point is valid
		gpose.pose.position.x = xn;
		gpose.pose.position.y = yn;
		navigation_features::PoseValid pv;
		pv.request.pose = gpose;
		if(!valid_client_.call(pv))
		{
			ROS_ERROR("record_bag_trajectories. Error calling service 'is_pose_valid'");
			return false;
		}
		ok = pv.response.ok;

	} while(!ok && i<2000);  

	if(!ok)
		return false;	

	printf("New GOAL position generated. frame:%s, x:%.2f, y:%.2f, th:%.2f\n\n", gpose.header.frame_id.c_str(), gpose.pose.position.x, gpose.pose.position.y, tf::getYaw(gpose.pose.orientation));

	//Publish a visualization marker with the goal
	visualization_msgs::Marker marker;
	marker.header.frame_id = gpose.header.frame_id;
	marker.header.stamp = ros::Time::now();
	marker.ns = "goal";
	marker.id = 10;
	marker.type = visualization_msgs::Marker::CYLINDER;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x = gpose.pose.position.x;
	marker.pose.position.y = gpose.pose.position.y;
	marker.pose.position.z = 0.2;
	marker.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);
	marker.scale.x = 0.3;
	marker.scale.y = 0.3;
	marker.scale.z = 0.1;
	marker.color.r = 1.0f;
	marker.color.g = 0.0f;
	marker.color.b = 0.0f;
	marker.color.a = 1.0;
	marker.lifetime = ros::Duration();
	goal_marker_pub_.publish(marker); 


	//Transform goal to the required frame
	//geometry_msgs::PoseStamped pose_out = transformPoseTo(gpose, store_frame_); 
	//gpose = pose_out;
		
	goalMutex_.lock();
	goal_ = gpose;
	goalMutex_.unlock();

	return true;
}



void peopleCallback(const upo_msgs::PersonPoseArrayUPO::ConstPtr& msg)
{
	ROS_INFO_ONCE("PeopleCallback. people received!");
	upo_msgs::PersonPoseArrayUPO aux = *msg;
	upo_msgs::PersonPoseArrayUPO aux2 = aux;
	
	aux2.header.frame_id = store_frame_;
	aux2.personPoses.clear();
	for(unsigned int p=0; p<aux.size; p++)
	{
		upo_msgs::PersonPoseUPO person = aux.personPoses.at(p);
		geometry_msgs::PoseStamped person_pose;
		person_pose.header = person.header;
		person_pose.pose.position = person.position;
		person_pose.pose.orientation = person.orientation;
		geometry_msgs::PoseStamped pose_out = transformPoseTo(person_pose, store_frame_); 
		person.header.frame_id = pose_out.header.frame_id;
		person.position = pose_out.pose.position;
		person.orientation = pose_out.pose.orientation;
		aux2.personPoses.push_back(person);
	}
	
	peopleMutex_.lock();
	people_ = aux2;
	peopleMutex_.unlock();
}



void pcCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
	ROS_INFO_ONCE("PcCallback. pointCloud received!");
	//if(recording_) 
	//{
		sensor_msgs::PointCloud2 lcloud;
		sensor_msgs::PointCloud2 in = *msg;
		//in.header.stamp = ros::Time();
		try{  
			if(!pcl_ros::transformPointCloud(store_frame_.c_str(), in, lcloud, *tf_))
			ROS_WARN("TransformPointCloud failed!!!!!");
		} catch (tf::TransformException ex){
			ROS_WARN("pcCallback. TransformException: %s", ex.what());
		}
	
		pcMutex_.lock();
		pc_ = lcloud;
		pcMutex_.unlock();
	//}
}




void velCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
	ROS_INFO_ONCE("velCallback. cmd_vel received!");
	velMutex_.lock();
	cmd_vel_ = *msg;
	velMutex_.unlock();
}





void storeRRTTrajectory(vector<geometry_msgs::PoseStamped>* path, string name)
{
	//Transform the path to the required frame
	vector<geometry_msgs::PoseStamped> p;
	for(unsigned int i=0; i<path->size(); i++)
	{
		geometry_msgs::PoseStamped pose = path->at(i);
		pose.header.stamp = ros::Time();
		geometry_msgs::PoseStamped out = transformPoseTo(pose, store_frame_);
		p.push_back(out);
	}

	//Take the goal and transform it
	goalMutex_.lock();
	geometry_msgs::PoseStamped gpose = goal_;
	goalMutex_.unlock();
	geometry_msgs::PoseStamped goal = transformPoseTo(gpose, store_frame_);

	// if the final point of the path is far from the goal
	// we discard this scenario
	float px = p[p.size()-1].pose.position.x;
	float py = p[p.size()-1].pose.position.y;
	float gx = goal.pose.position.x; 
	float gy = goal.pose.position.y;
	float dist = sqrt((px-gx)*(px-gx) + (py-gy)*(py-gy));
	if(dist >= 0.5)
		return;

	//take the robot initial position in the map (if exists)
	robotMutex_.lock();
	geometry_msgs::PoseStamped rm = robot_map_current_;
	robotMutex_.unlock();

	//Take the people 
	peopleMutex_.lock();
	upo_msgs::PersonPoseArrayUPO people = people_;
	peopleMutex_.unlock();


	//Take the obstacles
	pcMutex_.lock();
	sensor_msgs::PointCloud2 obs = pc_;
	pcMutex_.unlock();

	//Take the vels (not applicable in this case)

					
	//Write data in a bag file
	string n = store_dir_ + name + "-" + currentDateTime() + ".bag";
	rosbag::Bag traj_bag;
	traj_bag.open(n.c_str(), rosbag::bagmode::Write);
					
	ros::Time t = ros::Time::now();

	for(unsigned int j=0; j<p.size(); j++)
	{
		traj_bag.write("robot", t, p[j]);	//robot position 
		traj_bag.write("people", t, people); 	//people positions
		traj_bag.write("goal", t, goal); 		//goal 
		traj_bag.write("obstacles", t, obs); 	//obstacles
		//traj_bag.write("vels", t, vels); 		//robot vels
		traj_bag.write("robot_map", t, rm);	//robot position
	}

	traj_bag.close();		
} 





void storeStep()
{
	printf("...recording...\n");
	//take the robot position
	robotMutex_.lock();
	geometry_msgs::PoseStamped r = robot_current_;
	geometry_msgs::PoseStamped rm = robot_map_current_;
	robotMutex_.unlock();

	//Take the goal and transform it
	goalMutex_.lock();
	geometry_msgs::PoseStamped gpose = goal_;
	goalMutex_.unlock();
	geometry_msgs::PoseStamped goal = transformPoseTo(gpose, store_frame_);

	//Take the people 
	peopleMutex_.lock();
	upo_msgs::PersonPoseArrayUPO people = people_;
	peopleMutex_.unlock();

	//Take the obstacles
	pcMutex_.lock();
	sensor_msgs::PointCloud2 obs = pc_;
	pcMutex_.unlock();

	//Take the vels
	velMutex_.lock();
	geometry_msgs::Twist vels = cmd_vel_;
	velMutex_.unlock();

	path_.robot_data.push_back(r);
	path_.people_data.push_back(people);
	path_.goal_data.push_back(goal);
	path_.obs_data.push_back(obs);
	path_.vel_data.push_back(vels);
	path_.robot_map_data.push_back(rm);


	//When robot position is close to the goal, stop recording
	float goal_dist = sqrt(pow((goal.pose.position.x - r.pose.position.x), 2) + pow((goal.pose.position.y - r.pose.position.y), 2));
	if(goal_dist <= 0.15) {
		printf("Path recording ended. Path size: %i nodes.\n\n", (int)path_.robot_data.size());
		recording_ = false;
	}

}







int main(int argc, char** argv){

	ros::init(argc, argv, "record_bag_trajectories");
  	tf_ = new tf::TransformListener(ros::Duration(10));

	ros::NodeHandle n("~");
	ros::NodeHandle nh;
	
	n.param("use_rrt", use_rrt_path_, true);
	
	n.param("random_scenario", random_scenario_, true);

	n.param("store_dir", store_dir_, string(" "));

	n.param("store_frame", store_frame_, string("odom"));
	
	
	//Robot initial position (map coordinates) 
	//Used in case that random_scenario_ if false
	double i_x;
	n.param("initial_x", i_x, 0.0);
	double i_y;
	n.param("initial_y", i_y, 0.0);
	double i_theta;
	n.param("initial_theta", i_theta, 0.0);
	
	geometry_msgs::PoseStamped init;
	init.header.frame_id = "map";
	init.header.stamp = ros::Time::now();
	init.pose.position.x = i_x;
	init.pose.position.y = i_y;
	init.pose.position.z = 0.0;
	init.pose.orientation = tf::createQuaternionMsgFromYaw(i_theta);
	
	robot_current_ = init;


	//Used in case of random_scenario_ is true
	int robot_poses = 10;
	n.param("robot_poses", robot_poses, 10);
	int people_poses = 10;
	n.param("people_poses", people_poses, 10);
	int goal_poses = 10;
	n.param("goal_poses", goal_poses, 10);
	

	//Service clients
	valid_client_ = nh.serviceClient<navigation_features::PoseValid>("/navigation_features/is_pose_valid");
	plan_client_ = nh.serviceClient<upo_rrt_planners::MakePlan>("/RRT_ros_wrapper/makeRRTPlan");
	

	string people_topic;
	n.param("people_topic", people_topic, string("/people/navigation"));
	
	string robot_topic;
	n.param("robot_topic", robot_topic, string("/odom"));

	string goal_topic;
	n.param("goal_topic", goal_topic, string("/clicked_point"));

	string pc_topic;
	n.param("pointcloud_topic", pc_topic, string("/scan360/point_cloud"));

	string vels_topic;
	n.param("vels_topic", vels_topic, string("/cmd_vel"));
	
	//Topic subscriptions
	clicked_point_sub_ = nh.subscribe<geometry_msgs::PointStamped>(goal_topic.c_str(), 1, &clickedPointCallback);
	people_pose_sub_ = nh.subscribe<upo_msgs::PersonPoseArrayUPO>(people_topic.c_str(), 1, &peopleCallback); 
	odom_pose_sub_ = nh.subscribe<nav_msgs::Odometry>(robot_topic.c_str(), 1, &odomCallback);
	pc_sub_ = nh.subscribe<sensor_msgs::PointCloud2>(pc_topic.c_str(), 1, &pcCallback);
	vel_sub_ = nh.subscribe<geometry_msgs::Twist>(vels_topic.c_str(), 1, &velCallback);


	//Publications
	goal_marker_pub_ = nh.advertise<visualization_msgs::Marker>("recording_goal_marker", 1);
	initial_pose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 1);
	people_pose_pub_ = nh.advertise<upo_msgs::PersonPoseArrayUPO>("people_pose", 1);


	/*ros::spinOnce();
	//Publish robot initial position
	geometry_msgs::PoseWithCovarianceStamped new_pose;
	new_pose.header.frame_id = "map";
	new_pose.pose.pose = init.pose;
	new_pose.pose.covariance[0]=0.25;
	new_pose.pose.covariance[7]=0.25;
	new_pose.pose.covariance[35]=0.06853891945200942;
	ros::spinOnce();
	initial_pose_pub_.publish(new_pose);
	ros::spinOnce();*/

	//Read map data
	ros::ServiceClient map_client = nh.serviceClient<nav_msgs::GetMap>("/static_map");
	while (! ros::service::waitForService("/static_map",1)){
		ROS_INFO("Waiting for map service");
	}
	nav_msgs::GetMap srv;
	map_client.call(srv);
	ROS_INFO_STREAM(srv.response.map.info);
	//map_image_ = cv::Mat(srv.response.map.info.height, srv.response.map.info.width,CV_8UC1, cv::Scalar(0));
	nav_msgs::MapMetaData map_metadata = srv.response.map.info;
	resolution_ = (double) map_metadata.resolution; // m/cell
	width_ = map_metadata.width * resolution_;		// m
	height_ = map_metadata.height * resolution_;	// m
	origin_.push_back(map_metadata.origin.position.x);  // m
	origin_.push_back(map_metadata.origin.position.y);  // m
	origin_.push_back(tf::getYaw(map_metadata.origin.orientation)); // rad
 	//uint8_t *myData = map_image_.data;
	//ros::spinOnce();

	printf("...preparing system... wait for 11 seconds...\n");
	sleep(11);	

	printf("\n---------------------------------\n");
	printf("     RECORDING TRAJECTORIES  \n");
	printf("---------------------------------\n");

	
	
	//Record paths using the RRT*
	if(use_rrt_path_)
	{
		printf("Using RRT* to record paths\n\n");
			
		if(random_scenario_)
		{
			printf("Generating random scenarios..\n");
			int count = 0;
			//1.Set a random robot position
			for(unsigned int i=0; i<robot_poses; i++) 
			{
				printf("ROBOT iter %u\t",  (i+1));
				if(!randomNewRobotPose())
					continue;
				sleep(3);
				
				//2.Set a random number of person in random positions
				for(unsigned int j=0; j<people_poses; j++)
				{
					printf("PEOPLE iter %u-%u\t",(i+1),(j+1));
					if(!randomNewPeople())
						continue;
					sleep(1);

					//3.Set a random goal
					for(unsigned int k=0; k<goal_poses; k++)
					{
						printf("GOAL iter %u-%u-%u\t", (i+1), (j+1), (k+1));
						if(!randomNewGoal())
							continue;
						//sleep(1);
						//Call the service to plan with RRT
						upo_rrt_planners::MakePlan mp;
						goalMutex_.lock();
						mp.request.goal = goal_;
						goalMutex_.unlock();
						if(!plan_client_.call(mp))
						{
							ROS_ERROR("record_bag_trajectories. Error calling service 'makeRRTPlan'");
							return (-1);
						}
						bool ok = mp.response.ok;
						vector<geometry_msgs::PoseStamped> p = mp.response.path;

						ros::spinOnce();
						count++;
						char buf[10];
						sprintf(buf, "sc%i", count);
						string sc = string(buf);
						storeRRTTrajectory(&p, sc);
						//ros::spinOnce();							
					}
				}
			}


		} else { //No random scenario

			int count = 0;
			while(n.ok()) 
			{
				ros::Rate r(1.0);
				while(!recording_) {
					printf("Waiting to receive a new goal\n");
					ros::spinOnce();
					r.sleep();
				}
				
				
				//Call the service to plan with RRT
				ros::spinOnce();
				upo_rrt_planners::MakePlan mp;
				goalMutex_.lock();
				mp.request.goal = goal_;
				goalMutex_.unlock();
				if(!plan_client_.call(mp))
				{
					ROS_ERROR("record_random_rrt_paths. Error calling service 'makeRRTPlan'");
					return (-1);
				}
				bool ok = mp.response.ok;
				vector<geometry_msgs::PoseStamped> p = mp.response.path;

				count++;
				char buf[10];
				sprintf(buf, "sc%i", count);
				string sc = string(buf);
				storeRRTTrajectory(&p, sc);

				recording_ = false;
			}

		}

		
	} else  //record paths moving the robot with the joystick
	{
		printf("Using joystick control to record robot paths\n\n");
		
		int count = 0;
		while(n.ok())
		{
			ros::Rate r(1.0);
			while(!recording_) {
				printf("Waiting to receive a new goal\n");
				ros::spinOnce();
				r.sleep();
			}
			
			count++;
			printf("Recording path %i\n", count);
			ros::Rate loop_rate(10.0);
			while(n.ok() && recording_) 
			{
				ros::spinOnce();
				storeStep();
				loop_rate.sleep();
			}	
					
			//Write data in a bag file
			char buf[10];
			sprintf(buf, "sc%i", count);
			string sc = string(buf);
			string name = store_dir_ + sc + "-" + currentDateTime() + ".bag";
			rosbag::Bag traj_bag;
			traj_bag.open(name.c_str(), rosbag::bagmode::Write);
					
			ros::Time t = ros::Time::now();

			for(unsigned int i=0; i<path_.robot_data.size(); i++)
			{
				traj_bag.write("robot", t, path_.robot_data[i]);	//robot position 
				traj_bag.write("people", t, path_.people_data[i]); 	//people positions
				traj_bag.write("goal", t, path_.goal_data[i]); 		//goal 
				traj_bag.write("obstacles", t, path_.obs_data[i]); 	//obstacles
				traj_bag.write("vels", t, path_.vel_data[i]); 		//robot vels
				traj_bag.write("robot_map", t, path_.robot_map_data[i]);	//robot position in map
			}

			traj_bag.close();
					
			//Clear the vectors
			path_.robot_data.clear();
			path_.people_data.clear();
			path_.goal_data.clear();
			path_.obs_data.clear();
			path_.vel_data.clear();
	
				
		}
	}
	
	
  	return(0);
}




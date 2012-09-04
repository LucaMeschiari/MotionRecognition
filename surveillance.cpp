//
// surveillance.cpp
//
// https://github.com/LucaMeschiari/MotionRecognition
//
// Copyright (c) 2012 Luca Meschiari <meschial@tcd.ie>  <http://www.lucameschiari.com>.
//
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// See <http://www.gnu.org/licenses/> for a copy of the
// GNU General Public License.
//

#ifdef _CH_
#pragma package <opencv>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"
#include "utilities.h"

//Define constants
#define alpha 0.005
#define k 3


void update_running_gaussian_averages( IplImage *current_frame, IplImage *averages_image, IplImage *stan_devs_image )
{

	//Get data for manage the 8 bit image and float images
	int width_step_cur=current_frame->widthStep;
	int pixel_step_cur=current_frame->widthStep/current_frame->width;
	
	int width_step_avg=averages_image->widthStep;
	int pixel_step_avg=averages_image->widthStep/averages_image->width;
	int number_channels_avg=averages_image->nChannels;
	
	int row=0,col=0;
	
	//Scan every pixel of images
	for (row=0; row < current_frame->height; row++)
		for (col=0; col < current_frame->width; col++)
		{
			//Get current point of current_frame
			unsigned char* curr_point_current = GETPIXELPTRMACRO( current_frame, col, row, width_step_cur, pixel_step_cur );
			
			//Get current point of averages_image and stan_devs_image casting the char* returned by GETPIXELMACRO to float
			float* curr_point_avg=reinterpret_cast<float*>GETPIXELPTRMACRO( averages_image, col, row, width_step_avg, pixel_step_avg );
			float* curr_point_stdev=reinterpret_cast<float*>GETPIXELPTRMACRO( stan_devs_image, col, row, width_step_avg, pixel_step_avg );
			
			//Update the values for every channel
			for (int i=0;i<3;i++){
				//Calc the update of the standard deviation of the current point for the current channel
				curr_point_stdev[i]=sqrt((alpha * pow(((float)curr_point_current[i]-curr_point_avg[i]),2) + ((1-alpha)*pow(curr_point_stdev[i],2))));
				//Calc the update of the average of the current point for the current channel
				curr_point_avg[i]=(float)((alpha * ((double)curr_point_current[i])) + ((1-alpha)*((double)curr_point_avg[i])));
			}
			
			//Update the current point on the averages_image and the sta_devs_image with the new values, The float array pointer is casted to unsigned char pointer to use PUTPIXELMACRO
			PUTPIXELMACRO(averages_image,col,row,(unsigned char*)curr_point_avg,width_step_avg, pixel_step_avg,number_channels_avg);
			PUTPIXELMACRO(stan_devs_image,col,row,(unsigned char*)curr_point_stdev,width_step_avg, pixel_step_avg,number_channels_avg);
		}
}

void determine_moving_points_using_running_gaussian_averages( IplImage *current_frame, IplImage *averages_image, IplImage *stan_devs_image, IplImage *moving_mask_image )
{


	//Get data for manage the 8 bit image and float images
	int width_step_cur=current_frame->widthStep;
	int pixel_step_cur=current_frame->widthStep/current_frame->width;
	int number_channels_cur=current_frame->nChannels;
	
	int width_step_avg=averages_image->widthStep;
	int pixel_step_avg=averages_image->widthStep/averages_image->width;

	int row=0,col=0;

	//Clear mask image
	cvZero(moving_mask_image);

	//Scan every pixel of images
	for (row=0; row < current_frame->height; row++)
		for (col=0; col < current_frame->width; col++)
		{
			//Get current point of current_frame
			unsigned char* curr_point_current = GETPIXELPTRMACRO( current_frame, col, row, width_step_cur, pixel_step_cur );

			//Get current point of averages_image and stan_devs_image casting the char* returned by GETPIXELMACRO to float
			float* curr_point_avg=reinterpret_cast<float*>GETPIXELPTRMACRO( averages_image, col, row, width_step_avg, pixel_step_avg );
			float* curr_point_stdev=reinterpret_cast<float*>GETPIXELPTRMACRO( stan_devs_image, col, row, width_step_avg, pixel_step_avg );

			float abs_diff[3];
			float stdev_mul[3];

			unsigned char white_pixel[4] = {255,255,255};

			//Check every channel
			for (int i=0; i<3; i++){
				//Calc the absolute difference
				abs_diff[i]=(float)fabs(((double)curr_point_current[i])-((double)curr_point_avg[i]));
				//Calc k*standard deviation
				stdev_mul[i]=curr_point_stdev[i]*k;

			}

			//Check if (abs_diff> k*std_dev) in at leat one channel
			if((abs_diff[0]>stdev_mul[0])||(abs_diff[1]>stdev_mul[1])||(abs_diff[2]>stdev_mul[2])){
				//Put white pixel in the moving_mask_image  -->The moving mask image will be a binary image and i can use opening and closing on it to clean the result
				PUTPIXELMACRO( moving_mask_image, col, row, white_pixel, width_step_cur, pixel_step_cur, number_channels_cur );
			}

		}

		// Apply morphological opening and closing operations to clean up the image
		cvMorphologyEx( moving_mask_image, moving_mask_image, NULL, NULL, CV_MOP_OPEN, 1 );
		cvMorphologyEx( moving_mask_image, moving_mask_image, NULL, NULL, CV_MOP_CLOSE, 2 );
		
}


int main( int argc, char** argv )
{
    IplImage *current_frame=NULL;
	IplImage *running_average_background=NULL;

	IplImage *static_background_image=NULL;
	IplImage *static_moving_mask_image=NULL;
	IplImage *running_average_background_image=NULL;
	IplImage *running_average_moving_mask_image=NULL;
	IplImage *running_gaussian_average_background_average=NULL;
	IplImage *running_gaussian_average_background_sd=NULL;
	IplImage *running_gaussian_average_sd_image=NULL;
	IplImage *running_gaussian_average_background_image=NULL;
	IplImage *running_gaussian_average_moving_mask_image=NULL;

	IplImage *change_and_remain_changed_background_image=NULL;
	IplImage *subtracted_image=NULL;
	IplImage *moving_mask_image=NULL;

    int user_clicked_key=0;
	int show_ch = 'm';
	bool paused = false;
    
    // Load the video (AVI) file
    CvCapture *capture = cvCaptureFromAVI( " " );   //Add here the inputh video path
    // Ensure AVI opened properly
    if( !capture )
		return 1;    
    
    // Get Frames Per Second in order to playback the video at the correct speed
    int fps = ( int )cvGetCaptureProperty( capture, CV_CAP_PROP_FPS );
    
	// Explain the User Interface
    printf( "Hot keys: \n"
		    "\tESC - quit the program\n"
            "\tSPACE - pause/resume the video\n");

	// Create display windows for images
	cvNamedWindow( "Input video", 0 );
    cvNamedWindow( "Static Background", 0 );
    cvNamedWindow( "Running Average Background", 0 );
    cvNamedWindow( "Running Gaussian Average Background", 0 );
    cvNamedWindow( "Running Gaussian Average Stan. Dev.", 0 );
    cvNamedWindow( "Moving Points - Static", 0 );
    cvNamedWindow( "Moving Points - Running Average", 0 );
    cvNamedWindow( "Moving Points - Running Gaussian Average", 0 );

	// Setup mouse callback on the original image so that the user can see image values as they move the
	// cursor over the image.
    cvSetMouseCallback( "Input video", on_mouse_show_values, 0 );
	window_name_for_on_mouse_show_values="Input video";

    while( user_clicked_key != ESC ) {
		// Get current video frame
        current_frame = cvQueryFrame( capture );
        if( !current_frame ) // No new frame available
			break;
		image_for_on_mouse_show_values = current_frame; // Assign image for mouse callback
		cvShowImage( "Input video", current_frame );

		if (static_background_image == NULL)
		{	// The first time around the loop create the images for processing
			// General purpose images
			subtracted_image = cvCloneImage( current_frame );
			// Static backgound images
			static_background_image = cvCloneImage( current_frame );
			static_moving_mask_image = cvCreateImage( cvGetSize(current_frame), 8, 3 );
			cvShowImage( "Static Background", static_background_image );
			// Running average images
			running_average_background = cvCreateImage( cvGetSize(current_frame), IPL_DEPTH_32F, 3 );
			//cvZero(running_average_background);
			cvConvert(current_frame, running_average_background);
			running_average_background_image = cvCloneImage( current_frame );
			running_average_moving_mask_image = cvCreateImage( cvGetSize(current_frame), 8, 3 );
			// Running Gaussian average images
			running_gaussian_average_background_image = cvCloneImage( current_frame );
			running_gaussian_average_sd_image = cvCloneImage( current_frame );
			running_gaussian_average_moving_mask_image = cvCreateImage( cvGetSize(current_frame), 8, 3 );
			running_gaussian_average_background_average = cvCreateImage( cvGetSize(current_frame), IPL_DEPTH_32F, 3 );
			cvConvert(current_frame, running_gaussian_average_background_average);
			running_gaussian_average_background_sd = cvCreateImage( cvGetSize(current_frame), IPL_DEPTH_32F, 3 );
			cvZero(running_gaussian_average_background_sd);
		}
		// Static Background Processing
		cvAbsDiff( current_frame, static_background_image, subtracted_image );
		cvThreshold( subtracted_image, static_moving_mask_image, 30, 255, CV_THRESH_BINARY );
        cvShowImage( "Moving Points - Static", static_moving_mask_image );

		// Running Average Background Processing
		cvRunningAvg( current_frame, running_average_background, 0.01 /*, moving_mask_image*/ );
		cvConvert( running_average_background, running_average_background_image );
		cvAbsDiff( current_frame, running_average_background_image, subtracted_image );
		cvThreshold( subtracted_image, running_average_moving_mask_image, 30, 255, CV_THRESH_BINARY );
		cvShowImage( "Running Average Background", running_average_background_image );
        cvShowImage( "Moving Points - Running Average", running_average_moving_mask_image );
		
		
		// Running Gaussian Average Background Processing
		
		update_running_gaussian_averages( current_frame, running_gaussian_average_background_average, running_gaussian_average_background_sd );
		cvConvertScaleAbs( running_gaussian_average_background_average, running_gaussian_average_background_image, 1.0, 0 );
		cvShowImage( "Running Gaussian Average Background", running_gaussian_average_background_image );
		cvConvertScaleAbs( running_gaussian_average_background_sd, running_gaussian_average_sd_image, 10.0, 0 );
		cvShowImage( "Running Gaussian Average Stan. Dev.", running_gaussian_average_sd_image );
		determine_moving_points_using_running_gaussian_averages( current_frame, running_gaussian_average_background_average, running_gaussian_average_background_sd, running_gaussian_average_moving_mask_image );
        cvShowImage( "Moving Points - Running Gaussian Average", running_gaussian_average_moving_mask_image );

        // Deal with user input, and wait for the delay between frames
		do {
			if( user_clicked_key == ' ' )
			{
				paused = !paused;
			}
			if (paused)
				user_clicked_key = cvWaitKey(0);
			else user_clicked_key = cvWaitKey( 1000 / fps );
		} while (( user_clicked_key != ESC ) && ( user_clicked_key != -1 ));
	}
    
    /* free memory */
    cvReleaseCapture( &capture );
 	cvDestroyWindow( "Input video" );
    cvDestroyWindow( "Static Background" );
    cvDestroyWindow( "Running Average Background" );
    cvDestroyWindow( "Running Gaussian Average Background" );
    cvDestroyWindow( "Running Gaussian Average Stan. Dev." );
    cvDestroyWindow( "Moving Points - Static" );
    cvDestroyWindow( "Moving Points - Running Average" );
    cvDestroyWindow( "Moving Points - Running Gaussian Average" );

    return 0;
}
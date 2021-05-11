#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;

int main() {

	Mat img1 = imread("img1.jpg");
	Mat img2 = imread("img2.jpg");

	imshow("1", img1);
	//	imshow("2", img2);

	for (double canny_threshold : { 40.0, 90.0, 140.0 })
	{
		Mat canny_output;
		Canny(img1, canny_output, canny_threshold, 3.0 * canny_threshold, 3, true);

		const string name = "Canny Output Threshold " + to_string((size_t)canny_threshold);
		namedWindow(name, cv::WINDOW_AUTOSIZE);
		imshow(name, canny_output);
	}
	waitKey(0);

	// ---------------

	Mat canny_output;
	const double canny_threshold = 100.0;
	Canny(img1, canny_output, canny_threshold, 3.0 * canny_threshold, 3, true);

	typedef vector<Point> Contour;     // a single contour is a vector of many points
	typedef vector<Contour> VContours;  // many of these are combined to create a vector of contour points

	VContours contours;
	vector<Vec4i> hierarchy;
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (auto & c : contours)
	{
		cout << "contour area: " << cv::contourArea(c) << std::endl;
	}

	const Scalar green(0, 255, 0);
	Mat output = img1.clone();
	for (auto & c : contours)
	{
		polylines(output, c, true, green, 1, LINE_AA);
	}
	namedWindow("Contours Drawn Onto Image", cv::WINDOW_AUTOSIZE);
	imshow("Contours Drawn Over Image", output);
	waitKey(0);

	// ---------------

	Mat blurred_image;
	GaussianBlur(img1, blurred_image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	const size_t erosion_and_dilation_iterations = 3;

	Mat eroded;
	erode(blurred_image, eroded, cv::Mat(), cv::Point(-1, -1), erosion_and_dilation_iterations);

	Mat dilated;
	dilate(eroded, dilated, cv::Mat(), cv::Point(-1, -1), erosion_and_dilation_iterations);

	Canny(dilated, canny_output, canny_threshold, 3.0 * canny_threshold, 3, true);
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat better_output = img1.clone();
	for (auto & c : contours)
	{
		polylines(better_output, c, true, green, 1, cv::LINE_AA);
	}
	namedWindow("Another Attempt At Contours", cv::WINDOW_AUTOSIZE);
	imshow("Another Attempt At Contours", better_output);

	waitKey(0);
	return 0;
}
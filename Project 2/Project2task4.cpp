#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;

Mat ownBilateralFilter(Mat , int , double , double );

void implementBilateralFilter(Mat img, const char* name)
{
    // Applying the bilateral filter with varying values of sigmaI and sigmaS and calculating the time difference in seconds

    float sigmaI[] = {10, 100, 500};
    float sigmaS[] = {2, 100, 500};
    Mat filteredImageOwn;
    Mat ImageOpenCV;
    int k =1;
    for(int i=0; i<sizeof(sigmaI)/sizeof(sigmaI[0]); i++)
    {
        for(int j=0; j<sizeof(sigmaS)/sizeof(sigmaS[0]); j++)
        {
            char filename[30];
            char filenameCV[30];

            // OpenCV bilateral filter
            bilateralFilter(img, ImageOpenCV, 15, sigmaI[i], sigmaS[j]);
            sprintf (filenameCV, "Output%sCV%d.png", name, k);
            imwrite(filenameCV , ImageOpenCV);

            time_t start = time(NULL);
            // Own bilateral filter
            filteredImageOwn = ownBilateralFilter(img, 15, sigmaI[i], sigmaS[j]);
            time_t end = time(NULL);
            std::cout << "\n Time taken at sigma_C = "<< sigmaI[i]<<" and sigma_S = "<< sigmaS[j]<< " is: " <<(end - start)<<"\n";
            sprintf (filename, "Output%sTask%d.png", name, k);
            imwrite(filename , filteredImageOwn);
            k++;
        }
    }
    
}

int main(int argc, char** argv ) 
{
    // Main Function

    Mat img;
    img = imread( "clock.jpg", 0 );

    if ( !img.data )
    {
        std::cout << "\n Check image data";
        return -1;
    }
    implementBilateralFilter(img, "Clock");

    img = imread( "bauckhage.jpg", 0 );

    if ( !img.data )
    {
        std::cout << "\n Check image data";
        return -1;
    }
    implementBilateralFilter(img, "Bauckhage");


    return 0;
}

float euclideanDistance(int x, int y, int i, int j) 
{
    // Calculating distance
    return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double gaussian(float x, double sigma) 
{
    // Calculating Gaussian value.
    return exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));

}

Mat ownBilateralFilter(Mat src, int filterLength, double sigmaI, double sigmaS) 
{
    // Bilateral filter logic implementation.

    Mat filteredImage = Mat::zeros(src.rows,src.cols,CV_64F);
    int half = filterLength / 2;
    int width = src.cols;
    int height = src.rows;


    for(int x = 0; x < height ; x++) 

    {
        for(int y = 0; y < width ; y++) 
        {
            double FilteredIntensity = 0;
            double norm = 0;
            int neighborOfX = 0;
            int neighborOfY = 0;
            

            for(int i = 0; i < filterLength ; i++) 
            {
                for(int j = 0; j < filterLength ; j++) 
                {
                    neighborOfX = x - (half - i);
                    neighborOfY = y - (half - j);
                    double gaussI = gaussian(src.at<uchar>(neighborOfX, neighborOfY) - src.at<uchar>(x, y), sigmaI);
                    double gaussS = gaussian(euclideanDistance(x, y, neighborOfX, neighborOfY), sigmaS);
                    double w = gaussI * gaussS;
                    FilteredIntensity = FilteredIntensity + src.at<uchar>(neighborOfX, neighborOfY) * w;
                    norm = norm + w;
        }
    }
    FilteredIntensity = FilteredIntensity / norm;
    filteredImage.at<double>(x, y) = FilteredIntensity;

        }
    }
    return filteredImage;
}


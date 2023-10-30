#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std; 

int main(){
    Mat image1 = imread("image1.jpg");
    Mat image2 = imread("image2.jpg");

    if (image1.size() != image2.size()) {
        printf("Two images are not in the same size");
        return -1;
    }

    Mat img = image1; 
    imshow("Original", img);
    imshow("blue", rgb[0]);
    imshow("green", rgb[1]);
    imshow("red", rgb[2]);
    imwrite("blue.jpg", rgb[0]);
    imwrite("green.jpg", rgb[1]);
    imwrite("red.jpg", rgb[2]);

}

void imgYUV(Mat img) {
    Mat rgb[3];
    Mat y(img.rows, img.cols, CV_8UC1);
    Mat u(img.rows, img.cols, CV_8UC1);
    Mat v(img.rows, img.cols, CV_8UC1);
    split(img, rgb);
    int R, G, B, Y, U, V;13
    for (int i = 0 ; i<img.rows;i++){
        for (int j = 0; j < img.cols; j++)
        {
        R = rgb[2].at<uchar>(i, j);
        G = rgb[1].at<uchar>(i, j);
        B = rgb[0].at<uchar>(i, j);
        Y = 0.299 * R + 0.587 * G + 0.114 * B;
        U = 128 - 0.168736 * R - 0.331264 * G + 0.5
        * B;
        V = 128 + 0.5 * R - 0.418688 * G - 0.081312
        * B;
        y.at<uchar>(i, j) = Y;
        u.at<uchar>(i, j) = U;
        v.at<uchar>(i, j) = V;
    }}
    imshow("V", v);
    imshow("U", u);
    imshow("Y", y);
}

void VeBieuDoHistogram(Mat img, String nameWindow) {
    // Khoi tao gia tri
    int histSize = 256; // so luong pixel cho moi gia
    tri pixel (0-255)
    float range[] = { 0, 255 }; //Pham vi gia tri muon do
    const float *ranges[] = { range };
    // Tinh histogram
    MatND hist;
    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize,
    ranges, true, false);
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0,
    0));
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX,
    -1, Mat());
    // Vẽ biểu đồ Hist
    for (int i = 1; i < histSize; i++)
    {
    line(histImage, Point(bin_w*(i - 1), hist_h -
    cvRound(hist.at<float>(i - 1))),
    Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
    Scalar(255, 0, 0), 2, 8, 0);
    }
    imshow(nameWindow, histImage);
}

void Histogram(Mat img) {
    img = imread("Histogram.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    String namewindow = "Bieu do Hist truoc can bang";
    VeBieuDoHistogram(img, namewindow);
    Mat imgDst(img.rows, img.cols, CV_8UC3);
    equalizeHist(img, imgDst);
    imshow("Sau can bang Histogram", imgDst);
    String namewindow1 = "Bieu do Hist sau can bang";
    VeBieuDoHistogram(imgDst, namewindow1);
}

void insertionSort(int window[]) {
    int temp, i, j;
    for (i = 0; i < 9; i++) {
    temp = window[i];
    for (j = i - 1; j >= 0 && temp < window[j]; j--)
    {
    window[j + 1] = window[j];
    }
    window[j + 1] = temp;
    }
}

void medianFilter(Mat img){
    img=imread("medianfilter.png",CV_LOAD_IMAGE_GRAYSCALE);
    int window[9];
    Mat dst = img.clone();
    for (int y = 0; y < img.rows; y++)
    for (int x = 0; x < img.cols; x++)
    dst.at<uchar>(y, x) = 0.0;
    for (int y = 1; y < img.rows - 1; y++)
        for (int x = 1; x < img.cols - 1; x++)
            window[0] = img.at<uchar>(y - 1, x - 1);
            window[1] = img.at<uchar>(y, x - 1);
            window[2] = img.at<uchar>(y + 1, x - 1);
            window[3] = img.at<uchar>(y - 1, x);
            window[4] = img.at<uchar>(y, x);
            window[5] = img.at<uchar>(y + 1, x);
            window[6] = img.at<uchar>(y - 1, x + 1);
            window[7] = img.at<uchar>(y, x + 1);
            window[8] = img.at<uchar>(y + 1, x + 1);
    insertionSort(window);
    dst.at<uchar>(y, x) = window[4];
    imshow("Anh goc", img);
    imshow("Sau Median Filter", dst);
}

void meanFilter(Mat img){
    img=imread("medianfilter.png",CV_LOAD_IMAGE_GRAYSCALE);
    int sizeMatrix = 9;
    Mat dst = img.clone();
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            dst.at<uchar>(y, x) = 0.0;
            double mean = 0.0;
    for (int y = 1; y < img.rows - 1; y++)
        for (int x = 1; x < img.cols - 1; x++)
            mean = (img.at<uchar>(y - 1, x - 1) 
            + img.at<uchar>(y, x - 1) 
            + img.at<uchar>(y + 1, x - 1)
            + img.at<uchar>(y - 1, x)
            + img.at<uchar>(y, x)
            + img.at<uchar>(y + 1, x)
            + img.at<uchar>(y - 1, x + 1)
            + img.at<uchar>(y, x + 1)
            + img.at<uchar>(y + 1, x + 1))/sizeMatrix;
    dst.at<uchar>(y, x) = mean;
    imshow("Anh goc", img);
    imshow("Sau Mean Filter", dst);
}

void TachCanhSobel(Mat img) {
    img = imread("Lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    int Gx[9] = {-1, -5, -1, 0,0,0, 1,5,1};
    int Gy[9] = {-1,0,1,-5,0,5,-1,0,1};
    Mat dst = img.clone();
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            dst.at<uchar>(y, x) = 0.0;
            double AGx = 0.0, AGy = 0.0;
    for (int y = 1; y < img.rows - 1; y++) {
        for (int x = 1; x < img.cols - 1; x++) {
            AGx = img.at<uchar>(y - 1, x - 1) * Gx[0]
                + img.at<uchar>(y - 1, x) * Gx[1]
                + img.at<uchar>(y - 1, x + 1) * Gx[2]
                + img.at<uchar>(y, x - 1) * Gx[3]
                + img.at<uchar>(y, x) * Gx[4]
                + img.at<uchar>(y, x + 1) * Gx[5]
                + img.at<uchar>(y + 1, x - 1) * Gx[6]
                + img.at<uchar>(y + 1, x) * Gx[7]
                + img.at<uchar>(y + 1, x + 1) * Gx[8];
            AGy = img.at<uchar>(y - 1, x - 1)* Gy[0]
                + img.at<uchar>(y - 1, x) * Gy[1]
                + img.at<uchar>(y - 1, x + 1) * Gy[2]
                + img.at<uchar>(y, x - 1) * Gy[3]
                + img.at<uchar>(y, x) * Gy[4]
                + img.at<uchar>(y, x + 1) * Gy[5]
                + img.at<uchar>(y + 1, x - 1) * Gy[6]
                + img.at<uchar>(y + 1, x) * Gy[7]
                + img.at<uchar>(y + 1, x + 1)* Gy[8];
            double val = AGx + AGy;
            if (val > 127)
                val = 255;
            else
            val = 0;
        }
    }
    dst.at<uchar>(y, x) = val;
    imshow("Anh goc", img); imshow("Loc canh Sobel", dst);
}

void TachCanhCanny(Mat img) {
    img = imread("Lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Canny(img, img, 255 / 3, 255);
    imshow("Anh goc", img);
    imshow("Canny Filter", img);
}

void DCT(Mat img){
    img = imread("Lena.jpg");
    const int m = 8, n = 8;
    int val, matrix[m][n];
    printf("Ma tran block truoc DCT \n");
    for (int i = 0; i<m; i++)
    {
        for (int j = 0; j < n; j++)
            {
            matrix[i][j] = img.at<uchar>(i, j);
            printf("%d \t", matrix[i][j]);
            }
            printf("\n");
    }
    int i, j, k, l;
    float dct[m][n];
    float ci, cj, dct1, sum;
    Mat img_dct8(8, 8, CV_32F);
    printf("\nMa tran block sau DCT \n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (i == 0)
                ci = 1 / sqrt(m);
            else
                ci = sqrt(2) / sqrt(m);
            if (j == 0)
                cj = 1 / sqrt(n);
            else
                cj = sqrt(2) / sqrt(n);
                sum = 0;
            for (k = 0; k < m; k++) {
                for (l = 0; l < n; l++) {
                dct1 = matrix[k][l] *
                cos((2 * k + 1) * i * pi / (2 * m)) *
                cos((2 * l + 1) * j * pi / (2 * n));
                sum = sum + dct1;
                }
            }
            dct[i][j] = ci * cj * sum;
            img_dct8.at<float>(i, j) = dct[i][j];
            printf("%d\t", (int)dct[i][j]);
            }
        printf("\n");
    }
}

void GaussianFilter(Mat image) {
    image = imread("medianfilter.png");
    int rows = image.rows;
    int cols = image.cols;
    for (int i = 0; i <rows; i++)
    {
        Vec3b *ptr = image.ptr<Vec3b>(i);
        for (int j = 0; j < cols; j++)
        {
            Vec3b pixel = ptr[j];
        }
    }
    imshow("Truoc Gaussian Filter", image);
    Mat image_Gauss = image.clone();
    GaussianBlur(image, image_Gauss, Size(9, 9), 0, 0);
    for (int i = 0; i < rows; i++)
    {
        Vec3b *ptr = image_Gauss.ptr<Vec3b>(i);
        for (int j = 0; j < cols; j++)
        {
            Vec3b pixel = ptr[j];
        }
    }
    imshow("Sau Gaussian Filter:", image_Gauss);
}

void USM(Mat in, long size, float a, float thresh){
    in = imread("Lena.jpg");
    size += (1 - (size % 2));
    Mat inF32;
    in.convertTo(inF32, CV_32FC1);
    Mat out;
    GaussianBlur(inF32, out, cv::Size(size, size),0.0);
    Mat hp = inF32 - out;
    Mat hpabs = cv::abs(hp);
    Mat hpthr;
    threshold(hpabs, hpthr, thresh, 1.0, THRESH_BINARY);
    Mat ret1 = inF32 + a * hp.mul(hpthr);
    Mat ret;
    ret1.convertTo(ret, CV_8UC1);
    imshow("goc", in);
    imshow("Unsharp Mask", ret);
}

void Segmentation(Mat image){
    image = imread("shapes_and_colors.jpg");
    Mat image_gray(image.rows, image.cols, CV_8UC1);
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    Mat binary(image.rows, image.cols, CV_8UC1);
    threshold(image_gray, binary, 40, 255, THRESH_BINARY |
    THRESH_OTSU);
    Mat fg;
    erode(binary, fg, Mat(), Point(-1, -1), 2);
    Mat bg;
    dilate(binary, bg, Mat(), Point(-1, -1), 3);
    threshold(bg, bg, 1, 128, THRESH_BINARY_INV);
    Mat markers(binary.size(), CV_8U, Scalar(0));
    markers = fg + bg;
    markers.convertTo(markers, CV_32S);
    watershed(image, markers);
    markers.convertTo(markers, CV_8U);
    threshold(markers, markers, 40, 255, THRESH_BINARY |
    THRESH_OTSU);
    imshow("Org", image);
    imshow("Sau Segmentation", markers);
}
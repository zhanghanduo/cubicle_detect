//
// Created by hd on 1/18/18.
//
#include "darknet_ros/Blob.h"

namespace darknet_ros {

    Blob::Blob(float xmin, float ymin, float width, float height, unsigned long tr_id) {
        counter = 1;

        id = tr_id;

        cv::Rect currentBoundingRect = cv::Rect_<int>(static_cast<int>(xmin),
                                                      static_cast<int>(ymin),
                                                      static_cast<int>(width),
                                                      static_cast<int>(height));
        boundingRects.push_back(currentBoundingRect);

        dblCurrentDiagonalSize = currentBoundingRect.width * currentBoundingRect.width
                                 + currentBoundingRect.height * currentBoundingRect.height;


        cv::Point2f currentCenter;
        currentCenter.x = xmin + width/ 2;
        currentCenter.y = ymin + height / 2;
        centerPositions.push_back(currentCenter);

        blnStillBeingTracked = true;
        blnCurrentMatchFoundOrNewBlob = true;
        trackedInCurrentFrame = false;

        intNumOfConsecutiveFramesWithoutAMatch = 0;

        numOfConsecutiveFramesWithoutDetAsso = 0;

//        t_initialized = false;
//
//        UpdateAUKF(true);

        //-----------------KF----------------------------------
        /*int stateSize = 6, measSize = 4, contrSize = 0;
        kf = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);
        state = cv::Mat(stateSize, 1, CV_32F);  // [x,y,v_x,v_y,w,h];
        meas = cv::Mat(measSize, 1, CV_32F);    // [z_x,z_y,z_w,z_h]
        cv::setIdentity(kf.transitionMatrix);
        // Transition State Matrix A
        kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
        kf.measurementMatrix.at<float>(0) = 1.0f;
        kf.measurementMatrix.at<float>(7) = 1.0f;
        kf.measurementMatrix.at<float>(16) = 1.0f;
        kf.measurementMatrix.at<float>(23) = 1.0f;
        // Process Noise Covariance Matrix Q
        //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
        kf.processNoiseCov.at<float>(0) = 1e-2;
        kf.processNoiseCov.at<float>(7) = 1e-2;
        kf.processNoiseCov.at<float>(14) = 5.0f;
        kf.processNoiseCov.at<float>(21) = 5.0f;
        kf.processNoiseCov.at<float>(28) = 1e-2;
        kf.processNoiseCov.at<float>(35) = 1e-2;
        // Measures Noise Covariance Matrix R
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

        meas.at<float>(0) = xmin + width / 2;
        meas.at<float>(1) = ymin + height / 2;
        meas.at<float>(2) = width;
        meas.at<float>(3) = height;

        // >>>> Initialization
        kf.errorCovPre.at<float>(0) = 1; // px
        kf.errorCovPre.at<float>(7) = 1; // px
        kf.errorCovPre.at<float>(14) = 1;
        kf.errorCovPre.at<float>(21) = 1;
        kf.errorCovPre.at<float>(28) = 1; // px
        kf.errorCovPre.at<float>(35) = 1; // px

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(1) = meas.at<float>(1);
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = meas.at<float>(2);
        state.at<float>(5) = meas.at<float>(3);
        // <<<< Initialization

        kf.statePost = state;*/

        //--------------EKF-----------------------
//        double deltaT = 0.01, omega_w =8, omega_u = 3.1623;
//        EKF = cv::KalmanFilter(3, 2, 0);
//        cv::Mat_<float> measurement(2,1);
//        measurement.setTo(cv::Scalar(0));
//        EKF.statePost.at<float>(0) = 0; // X
//        EKF.statePost.at<float>(1) = 0; // dX
//        EKF.statePost.at<float>(2) = 0; // theta
//        EKF.transitionMatrix = (cv::Mat_<float>(3, 3) << 1,1,0,   0,1,0,  0,0,1  ); //f
//        EKF.measurementMatrix = (cv::Mat_<float>(2, 3) << 1,0,0, 0,0,1  );  //H
//        EKF.processNoiseCov = (cv::Mat_<float>(3, 3) << 1,0,0, 0,0.1,0, 0,0,0.1);
//        EKF.processNoiseCov *=pow(omega_w,2);
//        setIdentity(EKF.measurementNoiseCov, cv::Scalar::all(pow(omega_u,2)));
//        setIdentity(EKF.errorCovPost, cv::Scalar::all(50));
    }

    class AcceleratedModel: public cv::tracking::UkfSystemModel
    {
    public:
        AcceleratedModel(float deltaTime, bool rectModel)
                :
                cv::tracking::UkfSystemModel(),
                m_deltaTime(deltaTime),
                m_rectModel(rectModel)
        {

        }

        void stateConversionFunction(const cv::Mat& x_k, const cv::Mat& u_k, const cv::Mat& v_k, cv::Mat& x_kplus1)
        {
            float x0 = x_k.at<float>(0, 0);
            float y0 = x_k.at<float>(1, 0);
            float vx0 = x_k.at<float>(2, 0);
            float vy0 = x_k.at<float>(3, 0);
            float ax0 = x_k.at<float>(4, 0);
            float ay0 = x_k.at<float>(5, 0);

            x_kplus1.at<float>(0, 0) = x0 + vx0 * m_deltaTime + ax0 * (m_deltaTime*m_deltaTime) / 2;
            x_kplus1.at<float>(1, 0) = y0 + vy0 * m_deltaTime + ay0 * (m_deltaTime*m_deltaTime) / 2;
            x_kplus1.at<float>(2, 0) = vx0 + ax0 * m_deltaTime;
            x_kplus1.at<float>(3, 0) = vy0 + ay0 * m_deltaTime;
            x_kplus1.at<float>(4, 0) = ax0;
            x_kplus1.at<float>(5, 0) = ay0;

            if (m_rectModel)
            {
                x_kplus1.at<float>(6, 0) = x_k.at<float>(6, 0);
                x_kplus1.at<float>(7, 0) = x_k.at<float>(7, 0);
            }

            if (v_k.size() == u_k.size())
            {
                x_kplus1 += v_k + u_k;
            }
            else
            {
                x_kplus1 += v_k;
            }
        }

        void measurementFunction(const cv::Mat& x_k, const cv::Mat& n_k, cv::Mat& z_k)
        {
            float x0 = x_k.at<float>(0, 0);
            float y0 = x_k.at<float>(1, 0);
            float vx0 = x_k.at<float>(2, 0);
            float vy0 = x_k.at<float>(3, 0);
            float ax0 = x_k.at<float>(4, 0);
            float ay0 = x_k.at<float>(5, 0);

            z_k.at<float>(0, 0) = x0 + vx0 * m_deltaTime + ax0 * (m_deltaTime*m_deltaTime) / 2 + n_k.at<float>(0, 0);
            z_k.at<float>(1, 0) = y0 + vy0 * m_deltaTime + ay0 * (m_deltaTime*m_deltaTime) / 2 + n_k.at<float>(1, 0);

            if (m_rectModel)
            {
                z_k.at<float>(2, 0) = x_k.at<float>(6, 0);
                z_k.at<float>(3, 0) = x_k.at<float>(7, 0);
            }
        }

    private:
        float m_deltaTime;
        bool m_rectModel;
    };

    void Blob::CreateAugmentedUnscentedKF(cv::Rect_<float> rect0, cv::Point_<float> rectv0){

        int MP = 4;
        int DP = 8;
        int CP = 0;

        cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
        processNoiseCov.at<float>(0, 0) = 1e-3;
        processNoiseCov.at<float>(1, 1) = 1e-3;
        processNoiseCov.at<float>(2, 2) = 1e-3;
        processNoiseCov.at<float>(3, 3) = 1e-3;
        processNoiseCov.at<float>(4, 4) = 1e-3;
        processNoiseCov.at<float>(5, 5) = 1e-3;
        processNoiseCov.at<float>(6, 6) = 1e-3;
        processNoiseCov.at<float>(7, 7) = 1e-3;

        cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
        measurementNoiseCov.at<float>(0, 0) = 1e-3;
        measurementNoiseCov.at<float>(1, 1) = 1e-3;
        measurementNoiseCov.at<float>(2, 2) = 1e-3;
        measurementNoiseCov.at<float>(3, 3) = 1e-3;

        cv::Mat initState(DP, 1, Mat_t(1));
        initState.at<float>(0, 0) = rect0.x;
        initState.at<float>(1, 0) = rect0.y;
        initState.at<float>(2, 0) = rectv0.x;
        initState.at<float>(3, 0) = rectv0.y;
        initState.at<float>(4, 0) = 0;
        initState.at<float>(5, 0) = 0;
        initState.at<float>(6, 0) = rect0.width;
        initState.at<float>(7, 0) = rect0.height;

        cv::Mat P = 1e-3 * cv::Mat::eye(DP, DP, Mat_t(1));

        cv::Ptr<AcceleratedModel> model(new AcceleratedModel(0.05f, true));
//        cv::tracking::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
        cv::tracking::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
        params.dataType = Mat_t(1);
        params.stateInit = initState.clone();
        params.errorCovInit = P.clone();
        params.measurementNoiseCov = measurementNoiseCov.clone();
        params.processNoiseCov = processNoiseCov.clone();

        params.alpha = 1;
        params.beta = 2.0;
        params.k = -2.0;

//        uncsentedKF = cv::tracking::createAugmentedUnscentedKalmanFilter(params);
        uncsentedKF = cv::tracking::createUnscentedKalmanFilter(params);

        t_initialized = true;
    }

    cv::Rect Blob::GetRectPrediction() {
        if (t_initialized) {
            cv::Mat prediction;
            prediction = uncsentedKF->predict();

//            std::cout<<prediction<<std::endl;

            t_lastRectResult = cv::Rect_<float>(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(6), prediction.at<float>(7));
        }

        return cv::Rect(static_cast<int>(t_lastRectResult.x), static_cast<int>(t_lastRectResult.y), static_cast<int>(t_lastRectResult.width), static_cast<int>(t_lastRectResult.height));
    }

    cv::Rect Blob::UpdateAUKF(bool dataCorrect){

        cv::Rect currentBoundingRect = boundingRects.back();

        if (!t_initialized) {
            if (t_initialRects.size() < MIN_INIT_VALS) {
                if (dataCorrect)                {
                    t_initialRects.push_back(currentBoundingRect);
                }
            }
            if (t_initialRects.size() == MIN_INIT_VALS) {
                std::vector<cv::Point_<float> > initialPoints;
                cv::Point_<float> averageSize(0, 0);
                for (const auto& r : t_initialRects)
                {
                    initialPoints.push_back(cv::Point_<float>(r.x, r.y));
                    averageSize.x += r.width;
                    averageSize.y += r.height;
                }
                averageSize.x /= MIN_INIT_VALS;
                averageSize.y /= MIN_INIT_VALS;

                float kx = 0;
                float bx = 0;
                float ky = 0;
                float by = 0;

                float m1 = 0;  float m2 = 0;
                float m3_x = 0;  float m4_x = 0;
                float m3_y = 0;  float m4_y = 0;

                const float el_count = static_cast<float>(MIN_INIT_VALS - 0);
                for (size_t i = 0; i < MIN_INIT_VALS; ++i)
                {
                    m1 += i;
                    m2 += i*i;

                    m3_x += initialPoints[i].x;
                    m4_x += i * initialPoints[i].x;

                    m3_y += initialPoints[i].y;
                    m4_y += i * initialPoints[i].y;
                }
                float det_1 = static_cast<float>(1. / (el_count * m2 - m1 * m1));

                m1 *= -1.;

                kx = det_1 * (m1 * m3_x + el_count * m4_x);
                bx = det_1 * (m2 * m3_x + m1 * m4_x);

                ky = det_1 * (m1 * m3_y + el_count * m4_y);
                by = det_1 * (m2 * m3_y + m1 * m4_y);

//                get_lin_regress_params(initialPoints, 0, MIN_INIT_VALS, kx, bx, ky, by);
                cv::Rect_<float> rect0(kx * (MIN_INIT_VALS - 1) + bx, ky * (MIN_INIT_VALS - 1) + by, averageSize.x, averageSize.y);
                cv::Point_<float> rectv0(kx, ky);

                CreateAugmentedUnscentedKF(rect0, rectv0);

            }
        }

        if (t_initialized) {
            cv::Mat measurement(4, 1, Mat_t(1));
            if (!dataCorrect) {
                measurement.at<float>(0) = t_lastRectResult.x;  // update using prediction
                measurement.at<float>(1) = t_lastRectResult.y;
                measurement.at<float>(2) = t_lastRectResult.width;
                measurement.at<float>(3) = t_lastRectResult.height;
            } else {
                measurement.at<float>(0) = static_cast<float>(currentBoundingRect.x);//static_cast<float>(rect.x);  // update using measurements
                measurement.at<float>(1) = static_cast<float>(currentBoundingRect.y); //static_cast<float>(rect.y);
                measurement.at<float>(2) = static_cast<float>(currentBoundingRect.width); //static_cast<float>(rect.width);
                measurement.at<float>(3) = static_cast<float>(currentBoundingRect.height); //static_cast<float>(rect.height);
            }

            // Correction
            cv::Mat estimated;
            estimated = uncsentedKF->correct(measurement);

            t_lastRectResult.x = estimated.at<float>(0);   //update using measurements
            t_lastRectResult.y = estimated.at<float>(1);
            t_lastRectResult.width = estimated.at<float>(6);
            t_lastRectResult.height = estimated.at<float>(7);

        } else {
            t_lastRectResult.x = currentBoundingRect.x;//static_cast<float>(rect.x);
            t_lastRectResult.y = currentBoundingRect.y;//static_cast<float>(rect.y);
            t_lastRectResult.width = currentBoundingRect.width;//static_cast<float>(rect.width);
            t_lastRectResult.height = currentBoundingRect.height;//static_cast<float>(rect.height);
        }

        return cv::Rect(static_cast<int>(t_lastRectResult.x), static_cast<int>(t_lastRectResult.y), static_cast<int>(t_lastRectResult.width), static_cast<int>(t_lastRectResult.height));
    }

    void Blob::predictNextPosition() {

        auto vectorSize = static_cast<int>(centerPositions.size());
        auto numPositions = std::min(vectorSize,5);

        float deltaX = 0, deltaY=0;
        float sumOfXChanges = 0, sumOfYChanges =0, sumOfWeights =0;

        for (int ii=1; ii<numPositions; ii++) {
            int vecPosition = vectorSize - (numPositions-ii);
            sumOfXChanges += ii*(centerPositions[vecPosition].x - centerPositions[vecPosition-1].x);
            sumOfYChanges += ii*(centerPositions[vecPosition].y - centerPositions[vecPosition-1].y);
            sumOfWeights += ii;
        }

        if (vectorSize>1){
            deltaX = sumOfXChanges / sumOfWeights;//(int) std::round(sumOfXChanges / sumOfWeights);
            deltaY = sumOfYChanges / sumOfWeights;//(int) std::round(sumOfYChanges / sumOfWeights);
        }

        predictedNextPositionf.x = centerPositions.back().x + deltaX * (intNumOfConsecutiveFramesWithoutAMatch + 1);
        predictedNextPositionf.y = centerPositions.back().y + deltaY * (intNumOfConsecutiveFramesWithoutAMatch + 1);

        predictedNextPosition.x = static_cast<int>(predictedNextPositionf.x);
        predictedNextPosition.y = static_cast<int>(predictedNextPositionf.y );

    }

    void Blob::predictWidthHeight() {

        auto vectorSize = static_cast<int>(boundingRects.size());
        auto numPositions = std::min(vectorSize,5);

        float deltaX = 0.0f, deltaY=0.0f;
        int sumOfXChanges = 0, sumOfYChanges =0, sumOfWeights =0;

        for (int ii=1; ii<numPositions; ii++) {
            int vecPosition = vectorSize - (numPositions-ii);
            sumOfXChanges += ii*(boundingRects[vecPosition].width - boundingRects[vecPosition-1].width);
            sumOfYChanges += ii*(boundingRects[vecPosition].height - boundingRects[vecPosition-1].height);
            sumOfWeights += ii;
        }

        if (vectorSize>1){
            deltaX = (float)sumOfXChanges / sumOfWeights;
            deltaY = (float)sumOfYChanges / sumOfWeights;
        }

        predictedWidth = boundingRects.back().width + deltaX*(intNumOfConsecutiveFramesWithoutAMatch+1);
        predictedHeight = boundingRects.back().height + deltaY*(intNumOfConsecutiveFramesWithoutAMatch+1);

    }

}

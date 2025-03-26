
// C++实现RANSAC的二次曲线拟合
int ransac_curve_fitting(
    std::vector<float4> &in_cloud, std::vector<double> &best_model,
     std::vector<float4> &inliers, int maxiter=200, int mSamples=3, int min_inliers=10,
      float residual_thres=0.2)
{
    int point_num = in_cloud.size();
    std::default_random_engine rng;
    std::uniform_int_distribution<unsigned> uniform(0, point_num-1);
    rng.seed(10);

    // y = ax^2 + bx + c    <<<--- A_(0,0)=a, A_(1,0)=b, A_(2,0)=c --->>>
    Eigen::MatrixXd X_(3, 3), Y_(3, 1), A_(3, 1), points_x(point_num, 3), points_y(point_num, 1); //  dynamic cols.rows,
    for (int i = 0; i < point_num; ++i)
    {
      points_x(i, 0) = in_cloud[i].x * in_cloud[i].x;
      points_x(i, 1) = in_cloud[i].x;
      points_x(i, 2) = 1;
      points_y(i, 0) = in_cloud[i].y;
    }

    std::vector<unsigned int> selectIndexs;
    int best_inilers = 0;
    float best_error = 100.0;
    int iter = 0;
    int best_iter = 0;
    float tmp_error = 0.0;
    int num = 0;

    while (iter < maxiter)
    {
        selectIndexs.clear();
        inliers.clear();
        // 随机选n个点
        while (1)
        {
            unsigned int index = uniform(rng);
            selectIndexs.push_back(index);
            if(selectIndexs.size() == mSamples) // sample==2
            {
                break;
            }
        }
        // 模型参数估计
        for (size_t i = 0; i < selectIndexs.size(); ++i)
        {
          // std::cerr << selectIndexs[i] << std::endl;
          X_(i, 0) = in_cloud[selectIndexs[i]].x * in_cloud[selectIndexs[i]].x;
          X_(i, 1) = in_cloud[selectIndexs[i]].x;
          X_(i, 2) = 1;
          Y_(i, 0) = in_cloud[selectIndexs[i]].y;
        }
        try
        {
          X_.inverse();
        }
        catch(const std::exception& e)
        {
          std::cerr << e.what() << '\n';
          std::cerr << "Start the next loop..." << '\n';
          continue;
        }
        // X_为可逆方阵
        A_ = X_.inverse() * Y_;
        Eigen::MatrixXd y_pred = points_x * A_;
        Eigen::MatrixXd residual = points_y - y_pred;

        for (size_t i = 0; i < point_num; ++i)
        {
          if (abs(residual(i, 0)) < residual_thres)
          {
            inliers.push_back(in_cloud[i]);
          }

        }

        int inlier_num = inliers.size();
        if (inlier_num > best_inilers)
        {
          best_inilers = inlier_num;
          best_model[0] = A_(0,0);
          best_model[1] = A_(1,0);
          best_model[2] = A_(2,0);
          best_iter = iter;
        }

        if (inlier_num > min_inliers)
        {
          Eigen::MatrixXd better_model(3, 1), inliers_x(inlier_num, 3),
                          inliers_y(inlier_num, 1), y_pred_better(inlier_num, 1);
          for (size_t i = 0; i < inlier_num; ++i)
          {
            inliers_x(i, 0) = inliers[i].x * inliers[i].x;
            inliers_x(i, 1) = inliers[i].x;
            inliers_x(i, 2) = 1;
            inliers_y(i, 0) = inliers[i].y;
          }
          better_model = (inliers_x.transpose() * inliers_x).inverse() * inliers_x.transpose() * inliers_y;
          y_pred_better = inliers_x * better_model;
          float mean_square_error = 0;
          for (size_t i = 0; i < inlier_num; ++i)
          {
            mean_square_error += pow((y_pred_better(i,0) - inliers_y(i,0)), 2);
          }
          mean_square_error = mean_square_error / inlier_num;
          if (mean_square_error < best_error)
          {
            best_error = mean_square_error;
            best_model[0] = better_model(0,0);
            best_model[1] = better_model(1,0);
            best_model[2] = better_model(2,0);
            best_iter = iter;
          }

        }

        if (tmp_error != best_error)
        {
          tmp_error = best_error;
        }
        else
        {
          num += 1;
          if (num > 10)
          {
            break;
          }
        }
        std::cerr << "number of the error is constant: " << num << std::endl;
        std::cerr << "ransac iterations: " << iter << std::endl;
        iter++;
    }
    std::cerr << "best_iter: " << best_iter << "\n"
              << "best_model[0]: " << best_model[0] << "\n"
              << "best_error: " << best_error << "\n"
              << "inliers.size(): " << inliers.size() << std::endl;
    return 0;

}

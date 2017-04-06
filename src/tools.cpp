#include <cmath>
#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // RMSE = sqrt(sum((estimation - ground_truth)^2) / len(estimations)).
  // First, compute the sum of square difference.
  VectorXd square_diff = VectorXd(4);
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd estimation = estimations[i];
    VectorXd truth = ground_truth[i];
    for (int j = 0; j < square_diff.size(); ++j) {
      square_diff[j] += pow(estimation[j] - truth[j], 2);
    }
  }
  // Then, compute RMSE.
  VectorXd rmse = VectorXd(4);
  for (int j = 0; j < square_diff.size(); ++j) {
    rmse[j] = sqrt(square_diff[j] / estimations.size());
  }
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd result(3, 4);
  
  // Recover state parameters.
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  // Pre-compute a set of terms to avoid repeated calculation
  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = (c1 * c2);

  // Check division by zero
  if (fabs(c1) < 0.0001) {
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    return result;
  }

  // Compute the Jacobian matrix
  result << (px / c2), (py / c2), 0, 0,
            -(py / c1), (px / c1), 0, 0,
            py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3,
                px / c2, py / c2;

  return result;
}

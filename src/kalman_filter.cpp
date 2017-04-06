#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  // new state.
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd hx = VectorXd(3);
  hx[0] = sqrt(pow(x_[0], 2) + pow(x_[1], 2));
  hx[1] = atan(x_[1] / (x_[0] + 1e-6));
  if (hx[1] > M_PI) {
    hx[1] -= 2 * M_PI;
  } else if (hx[1] < - M_PI) {
    hx[1] += 2 * M_PI;
  }
  hx[2] = (x_[0] * x_[2] + x_[1] * x_[3]) / (hx[0] + 1e-6);

  VectorXd y = z - hx;
  
  MatrixXd Hj = tools_.CalculateJacobian(x_);
  MatrixXd Ht = Hj.transpose();
  MatrixXd S = Hj * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  
  // new state.
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  x_ = x_ + (K * y);
  P_ = (I - K * Hj) * P_;
}

#include "Eigen/Dense"
#include <bits/stdc++.h>
using namespace std;
using MatrixXf = Eigen::MatrixXf;
using RowVectorXf = Eigen::RowVectorXf;
using VectorXf = Eigen::VectorXf;

int main() {
    MatrixXf m1{{1, 2, 3}, {10, 20, 30}};
    RowVectorXf v{{100, -1000}};
    m1(0, 0) = 1000;
    cout << m1 << endl;
    cout << v << endl;
    cout << v * m1 << endl;
    v = v.array().max(0);
    cout << v.sum() << endl;
    VectorXf v1(10);
    cout << v1 << endl;
}
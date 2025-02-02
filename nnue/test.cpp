#include "Eigen/Dense"
#include <bits/stdc++.h>
using namespace std;
using MatrixXf = Eigen::MatrixXf;
using RowVectorXf = Eigen::RowVectorXf;

int main() {
    MatrixXf m1{{1, 2, 3}, {10, 20, 30}};
    RowVectorXf v{{100, 1000}};
    cout << m1 << endl;
    cout << v << endl;
    cout << v * m1 << endl;
}
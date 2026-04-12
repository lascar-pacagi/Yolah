#include <bits/stdc++.h>
#include <cstddef>

using namespace std;

int main(int argc, char *argv[]) {
  const string path1 = argv[1];
  const string path2 = argv[2];
  {
    ifstream ifs1(path1);
    ifstream ifs2(path2);
    double n = 0;
    double ok = 0;
    while (ifs1 && ifs2) {
      n++;
      double b1, d1, w1;
      ifs1 >> b1 >> d1 >> w1;
      int res1 = 0;
      double max1 = max({b1, d1, w1});
      if (d1 == max1)
        res1 = 1;
      else if (w1 == max1)
        res1 = 2;      
      double b2, d2, w2;
      ifs2 >> b2 >> d2 >> w2;
      int res2 = 0;
      double max2 = max({b2, d2, w2});
      if (d2 == max2)
        res2 = 1;
      else if (w2 == max2)
        res2 = 2;
      ok += res1 == res2;
    }
    cout << "accuracy: " << ok / n << "\n";
  }
  {
    ifstream ifs1(path1);
    ifstream ifs2(path2);
    double n = 0;
    double ok = 0;
    while (ifs1 && ifs2) {
      n++;
      double b1, d1, w1;
      ifs1 >> b1 >> d1 >> w1;
      double res1 = b1 - w1;
      double b2, d2, w2;
      ifs2 >> b2 >> d2 >> w2;
      double res2 = b2 - w2;
      if (res1 > 0 && res2 > 0)
        ok++;
      else if (res1 < 0 && res2 < 0)
        ok++;
      else if (res1 == 0 && res2 == 0)
        ok++;
    }
    cout << "sum accuracy: " << ok / n << "\n";
  }
}

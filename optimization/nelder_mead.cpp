#include "nelder_mead.h"
#include <random>
#include <chrono>
#include "indicators.h"
#include "misc.h"
#include <execution>
#include <iterator>
#include <sstream>
#include <iomanip>

using std::vector;

NelderMead::Builder& NelderMead::Builder::bounding_boxes(const vector<BoundingBox>& bb) {
    bounding_boxes_ = bb;
    return *this;
}

NelderMead::Builder& NelderMead::Builder::nb_iterations(size_t n) {
    nb_iterations_ = n;
    return *this;
}

NelderMead::Builder& NelderMead::Builder::reflexion(double v) {
    reflexion_ = v;
    return *this;
}

NelderMead::Builder& NelderMead::Builder::contraction(double v) {
    contraction_ = v;
    return *this;
}

NelderMead::Builder& NelderMead::Builder::expansion(double v) {
    expansion_ = v;
    return *this;
}

NelderMead::Builder& NelderMead::Builder::shrinkage(double v) {
    shrinkage_ = v;
    return *this;
}

NelderMead::Builder& NelderMead::Builder::fitness(FitnessFunction f) {
    fitness_ = f;
    return *this;
}

NelderMead NelderMead::Builder::build() const {
    return { bounding_boxes_, nb_iterations_, reflexion_, expansion_, contraction_, shrinkage_, fitness_ };
}

NelderMead::NelderMead(const vector<BoundingBox>& bounding_boxes, size_t nb_iterations, 
                       double reflexion, double expansion, double contraction, double shrinkage, FitnessFunction fitness) 
            : bounding_boxes(bounding_boxes), nb_iterations(nb_iterations), reflexion(reflexion), expansion(expansion),
              contraction(contraction), shrinkage(shrinkage), fitness(fitness) {
    std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    for (size_t i = 0; i <= bounding_boxes.size(); i++) {
        vector<double> w;
        for (const auto& bb : bounding_boxes) {
            std::uniform_real_distribution<double> d(bb.lo, bb.hi);
            w.push_back(d(generator));
        }
        weights.push_back(w);
    }
}

vector<double> NelderMead::best_weights() const {
    return weights.back();
}

namespace {
    vector<double> operator+(const vector<double>& pt1, const vector<double>& pt2) {
        vector<double> res(pt1.size());
        for (size_t i = 0; i < res.size(); i++) {
            res[i] = pt1[i] + pt2[i];
        }
        return res;
    }

    vector<double> operator*(double coeff, const vector<double>& pt) {
        vector<double> res(pt.size());
        for (size_t i = 0; i < res.size(); i++) {
            res[i] = coeff * pt[i];
        }
        return res;
    }

    vector<double> operator-(const vector<double>& pt1, const vector<double>& pt2) {
        return pt1 + (-1.0 * pt2);
    }
}

void NelderMead::run() {
    using namespace indicators;
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{""},
        option::ForegroundColor{Color::green},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };
    const size_t DIM = weights[0].size();
    const size_t NB_POINTS = DIM + 1;
    const size_t WORST = 0;
    const size_t SECOND_WORST = 1;
    const size_t BEST = DIM;
    vector<double> fitnesses(NB_POINTS);    
    auto compute_fitnesses = [&]{
        std::for_each(std::execution::par_unseq, begin(weights), end(weights), [&](vector<double>& w) {
            auto i = std::distance(&weights.data()[0], &w);
            fitnesses[i] = fitness(w);
        });
    };
    auto sort = [&]{
        sort_small(weights, fitnesses);
    };
    auto centroid = [&]{
        vector<double> res(DIM);
        for (size_t i = 1; i < NB_POINTS; i++) {
            for (size_t j = 0; j < DIM; j++) {
                res[j] += weights[i][j];
            }
        }
        for (auto& x : res) {
            x /= (NB_POINTS - 1);
        }
        return res;
    };
    compute_fitnesses();
    sort();
    for (size_t iter = 0; iter < nb_iterations; iter++) {
        std::stringbuf buf;
        std::ostream os(&buf);
        os << "Best value: " << fitnesses[BEST];
        os << " {" << weights[BEST][0];
        for (size_t i = 1; i < DIM; i++) {
            os << ", " << weights[BEST][i];
            os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << weights[BEST][i];
        }
        os << "}";
        bar.set_option(option::PostfixText{buf.str()});
        bar.set_progress(iter * 100 / nb_iterations);
        auto c = centroid();
        // Reflect
        auto worst_point = weights[WORST];
        auto reflection_point = c + reflexion * (c - worst_point);
        double v_reflection = fitness(reflection_point);
        if (fitnesses[SECOND_WORST] < v_reflection && v_reflection <= fitnesses[BEST]) {
            weights[WORST] = reflection_point;
            fitnesses[WORST] = v_reflection;
            sort();
            continue;
        }
        // Expand
        if (v_reflection > fitnesses[BEST]) {
            auto expansion_point = c + expansion * (reflection_point - c);
            double v_expansion = fitness(expansion_point);
            if (v_expansion > v_reflection) {
                weights[WORST] = expansion_point;
                fitnesses[WORST]  = v_expansion;
            } else {
                weights[WORST] = reflection_point;
                fitnesses[WORST]  = v_reflection;
            }
            sort();
            continue;
        }
        // Contract
        if (v_reflection <= fitnesses[SECOND_WORST]) {
            if (v_reflection >= fitnesses[WORST]) {
                // Outside
                auto outside_point = c + contraction * (reflection_point - c);
                double v_outside = fitness(outside_point);
                if (v_outside >= v_reflection) {
                    weights[WORST] = outside_point;
                    fitnesses[WORST] = v_outside;
                    sort();
                    continue;
                } 
            } else {
                // Inside
                auto inside_point = c + contraction * (worst_point - c);
                double v_inside = fitness(inside_point);
                if (v_inside >= fitnesses[WORST]) {
                    weights[WORST] = inside_point;
                    fitnesses[WORST] = v_inside;
                    sort();
                    continue;
                }
            }
        }
        // Shrink
        auto best = weights[BEST];
        for (size_t i = 0; i < NB_POINTS - 1; i++) {            
            weights[i] = best + shrinkage * (weights[i] - best);
        }
        compute_fitnesses();
        sort();
    }
    bar.set_progress(100);
}

void NelderMead::run_parallel(size_t nb_threads) {

}
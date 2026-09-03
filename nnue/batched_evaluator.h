#ifndef BATCHED_EVALUATOR_H
#define BATCHED_EVALUATOR_H
// batched_evaluator.h — merges the evaluate() calls of several search threads
// into larger batches before handing them to a backend.
//
// Why: a GPU (or the OpenMP CPU backend) is efficient only on large batches,
// whereas each MCTS thread can only gather a limited number of leaves per
// iteration without degrading the search (virtual-loss collisions). With T
// search threads each submitting ~B leaves, the backend sees batches of up to
// T·B positions instead of B.
//
// How: callers enqueue their (requests, results) spans and block. One worker
// thread wakes up as soon as
//   • the queued positions reach `max_batch`, or
//   • every registered client is waiting (`nb_clients`), or
//   • `max_wait` has elapsed with something queued,
// concatenates the queued requests, runs the backend once, scatters the
// results back and releases the callers. The queue is FIFO so no client can
// starve.
//
// An EMPTY request is a barrier join: the caller has nothing to evaluate but
// wants to block until the next batch has been processed (a search thread
// whose descents all collided with leaves owned by another thread). It counts
// as a waiting client, so it does not delay the batch.
#include "nn_evaluator.h"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nn {

class BatchingEvaluator final : public Evaluator {
public:
    // backend    : the evaluator that does the actual work (owned)
    // nb_clients : number of threads expected to call evaluate() concurrently;
    //              a batch is launched as soon as all of them are waiting
    // max_batch  : upper bound on the merged batch size
    // max_wait   : how long the worker waits for stragglers before launching
    //              a partial batch
    BatchingEvaluator(std::unique_ptr<Evaluator> backend, size_t nb_clients, size_t max_batch,
                      std::chrono::microseconds max_wait = std::chrono::microseconds(500));
    ~BatchingEvaluator() override;

    void evaluate(std::span<const Request> in, std::span<Result> out) override;
    size_t preferred_batch_size() const override { return max_batch; }
    std::string info() const override;
    void reload(const std::string& weights_filename) override;

    Evaluator& backend() { return *impl; }
    void set_nb_clients(size_t n);

private:
    struct Job {
        std::span<const Request> in;
        std::span<Result> out;
        bool done = false;
    };
    void worker_loop();

    std::unique_ptr<Evaluator> impl;
    size_t nb_clients;
    const size_t max_batch;
    const std::chrono::microseconds max_wait;

    std::mutex mutex;
    std::condition_variable cv_worker;     // signals the worker: new job / stop
    std::condition_variable cv_clients;    // signals clients: results ready
    std::deque<Job*> queue;
    size_t queued_positions = 0;
    size_t waiting_clients = 0;
    bool stop = false;
    std::thread worker;

    // Scratch owned by the worker.
    std::vector<Request> batch_in;
    std::vector<Result> batch_out;
};

} // namespace nn

#endif

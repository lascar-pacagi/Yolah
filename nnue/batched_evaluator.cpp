#include "batched_evaluator.h"
#include <algorithm>

namespace nn {

BatchingEvaluator::BatchingEvaluator(std::unique_ptr<Evaluator> backend, size_t nb_clients,
                                     size_t max_batch, std::chrono::microseconds max_wait)
    : impl(std::move(backend)), nb_clients(std::max<size_t>(1, nb_clients)),
      max_batch(std::max<size_t>(1, max_batch)), max_wait(max_wait) {
    batch_in.reserve(this->max_batch);
    batch_out.resize(this->max_batch);
    worker = std::thread([this] { worker_loop(); });
}

BatchingEvaluator::~BatchingEvaluator() {
    {
        std::lock_guard lock(mutex);
        stop = true;
    }
    cv_worker.notify_all();
    worker.join();
}

std::string BatchingEvaluator::info() const {
    return "batching(" + std::to_string(nb_clients) + " clients, max batch " +
           std::to_string(max_batch) + ") over " + impl->info();
}

void BatchingEvaluator::reload(const std::string& weights_filename) {
    // Serialise with the worker: no batch may be running while weights change.
    std::lock_guard lock(mutex);
    impl->reload(weights_filename);
}

void BatchingEvaluator::set_nb_clients(size_t n) {
    std::lock_guard lock(mutex);
    nb_clients = std::max<size_t>(1, n);
}

void BatchingEvaluator::evaluate(std::span<const Request> in, std::span<Result> out) {
    Job job{in, out};   // in.empty() = barrier join, see header
    std::unique_lock lock(mutex);
    queue.push_back(&job);
    queued_positions += in.size();
    ++waiting_clients;
    cv_worker.notify_one();
    cv_clients.wait(lock, [&] { return job.done; });
    --waiting_clients;
}

void BatchingEvaluator::worker_loop() {
    std::unique_lock lock(mutex);
    for (;;) {
        // Wake-up conditions (see header). wait_for handles the straggler timeout.
        auto ready = [&] {
            return stop || queued_positions >= max_batch ||
                   (!queue.empty() && waiting_clients >= nb_clients);
        };
        if (!ready()) {
            if (!queue.empty()) cv_worker.wait_for(lock, max_wait, ready);
            else                cv_worker.wait(lock, [&] { return stop || !queue.empty(); });
        }
        if (stop && queue.empty()) return;
        if (queue.empty()) continue;

        // Take whole jobs (FIFO) until the next one would overflow max_batch.
        // The first job is always taken, even if it alone exceeds max_batch:
        // backends accept any batch size, max_batch is only the merge target.
        std::vector<Job*> taken;
        batch_in.clear();
        while (!queue.empty()) {
            Job* j = queue.front();
            if (!taken.empty() && batch_in.size() + j->in.size() > max_batch) break;
            taken.push_back(j);
            batch_in.insert(batch_in.end(), j->in.begin(), j->in.end());
            queued_positions -= j->in.size();
            queue.pop_front();
        }
        if (batch_out.size() < batch_in.size()) batch_out.resize(batch_in.size());

        // Run the backend without holding the lock so clients can keep queueing.
        if (!batch_in.empty()) {
            lock.unlock();
            impl->evaluate(batch_in, std::span<Result>(batch_out.data(), batch_in.size()));
            lock.lock();
        }

        size_t offset = 0;
        for (Job* j : taken) {
            std::copy_n(batch_out.begin() + offset, j->in.size(), j->out.begin());
            offset += j->in.size();
            j->done = true;
        }
        cv_clients.notify_all();
    }
}

} // namespace nn

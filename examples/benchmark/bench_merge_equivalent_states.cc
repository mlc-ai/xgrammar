#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Build from the repository root after `cmake --build build`:
// g++ -O3 -std=c++17 -Icpp -Iinclude -I3rdparty/picojson -I3rdparty/dlpack/include \
//   examples/benchmark/bench_merge_equivalent_states.cc build/libxgrammar.a \
//   -o build/bench_merge_equivalent_states
#include "fsm.h"

using namespace xgrammar;

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t checksum = 0;

FSMWithStartEnd MakeNoMerge(int number_of_states) {
  FSMWithStartEnd fsm;
  for (int state = 0; state < number_of_states; ++state) {
    fsm.AddState();
  }
  fsm.SetStartState(0);
  fsm.AddEndState(number_of_states - 1);
  for (int state = 0; state < number_of_states; ++state) {
    fsm.GetFsm().AddRuleEdge(state, (state + 1) % number_of_states, state);
  }
  return fsm;
}

FSMWithStartEnd MakeOneRound(int number_of_branches) {
  FSMWithStartEnd fsm;
  int start_state = fsm.AddState();
  fsm.SetStartState(start_state);
  for (int branch = 0; branch < number_of_branches; ++branch) {
    int end_state = fsm.AddState();
    fsm.GetFsm().AddEdge(start_state, end_state, 'a', 'a');
    fsm.AddEndState(end_state);
  }
  return fsm;
}

FSMWithStartEnd MakeDeepPrefix(int number_of_branches, int depth) {
  FSMWithStartEnd fsm;
  int start_state = fsm.AddState();
  fsm.SetStartState(start_state);
  for (int branch = 0; branch < number_of_branches; ++branch) {
    int current_state = start_state;
    for (int level = 0; level < depth; ++level) {
      int next_state = fsm.AddState();
      int character = 'a' + level % 26;
      fsm.GetFsm().AddEdge(current_state, next_state, character, character);
      current_state = next_state;
    }
    fsm.AddEndState(current_state);
  }
  return fsm;
}

double Median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void Benchmark(const std::string& name, const FSMWithStartEnd& fsm, int calls_per_sample) {
  constexpr int kWarmupSamples = 3;
  constexpr int kMeasuredSamples = 15;

  auto RunCalls = [&] {
    for (int call = 0; call < calls_per_sample; ++call) {
      auto result = fsm.MergeEquivalentStates();
      checksum += result.NumStates();
    }
  };

  for (int sample = 0; sample < kWarmupSamples; ++sample) {
    RunCalls();
  }

  std::vector<double> nanoseconds_per_call;
  nanoseconds_per_call.reserve(kMeasuredSamples);
  for (int sample = 0; sample < kMeasuredSamples; ++sample) {
    auto begin = Clock::now();
    RunCalls();
    auto end = Clock::now();
    double elapsed_nanoseconds = std::chrono::duration<double, std::nano>(end - begin).count();
    nanoseconds_per_call.push_back(elapsed_nanoseconds / calls_per_sample);
  }

  auto merged = fsm.MergeEquivalentStates();
  std::cout << name << " states=" << fsm.NumStates() << "->" << merged.NumStates()
            << " median_us=" << std::fixed << std::setprecision(3)
            << Median(nanoseconds_per_call) / 1000.0 << '\n';
}

}  // namespace

int main() {
  Benchmark("small_no_merge_8", MakeNoMerge(8), 20000);
  Benchmark("small_no_merge_32", MakeNoMerge(32), 5000);
  Benchmark("small_no_merge_128", MakeNoMerge(128), 1000);
  Benchmark("small_one_round_8", MakeOneRound(7), 20000);
  Benchmark("small_one_round_32", MakeOneRound(31), 5000);
  Benchmark("small_one_round_128", MakeOneRound(127), 1000);
  Benchmark("small_deep_17", MakeDeepPrefix(4, 4), 5000);
  Benchmark("small_deep_65", MakeDeepPrefix(8, 8), 1000);

  Benchmark("large_no_merge_8000", MakeNoMerge(8000), 10);
  Benchmark("large_one_round_8000", MakeOneRound(7999), 10);
  Benchmark("large_deep_8001", MakeDeepPrefix(400, 20), 3);
  Benchmark("large_no_merge_80000", MakeNoMerge(80000), 1);
  Benchmark("large_one_round_80000", MakeOneRound(79999), 1);
  Benchmark("large_deep_80001", MakeDeepPrefix(4000, 20), 1);

  std::cout << "checksum=" << checksum << '\n';
  return 0;
}

//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "neural_network.h"
using namespace std;

static void test_1() {
    NeuralNetwork<float, Dense, Softmax> model1(
        0.5f,    // Dense<float>::weight
        -0.1f    // Dense<float>::bias
    );
    const std::vector<float> in1(5, 1.0f);
    auto out1 = model1.predict(in1);
    const auto total = std::accumulate(out1.begin(), out1.end(), 0.0);

    std::cout << out1.size() << std::endl;
    std::cout << total << std::endl;
    for (const auto& item: out1) { std::cout << item << ' '; }
    REQUIRE((out1.size() == 5));
}

TEST_CASE("Question #1.1") {
    execute_test("question_1_test_1.in", test_1);
}
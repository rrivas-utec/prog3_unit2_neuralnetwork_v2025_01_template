//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "neural_network.h"
using namespace std;

static void test_2() {
    NeuralNetwork<double, Dense, ReLU, Dense, Softmax> model2(
        0.2,   // Dense: weight - aplicable a todos los Dense
        0.0    // Dense: bias - aplicable a todos los Dense
    );

    std::vector<double> in2 = {1.0, 2.0, 3.0};
    auto out2 = model2.predict(in2);
    const auto total =std::accumulate(out2.begin(), out2.end(), 0.0);
    std::cout << out2.size() << std::endl;
    std::cout << total << std::endl;
    for (const auto& item: out2) { std::cout << item << ' '; }
    REQUIRE(out2.size() == 3);
}

TEST_CASE("Question #1.2") {
    execute_test("question_1_test_2.in", test_2);
}
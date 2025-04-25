//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "neural_network.h"
using namespace std;

static void test_3() {
    NeuralNetwork<float, Dense, Dropout, ReLU, Dense, Softmax> model3(
        0.05f,    // Dense: weight
        -0.02f    // Dense: bias
    );

    std::vector<float> in3 = {1,2,3,4,5,6};
    auto out3 = model3.predict(in3);
    REQUIRE(out3.size() == 3);
    for (const auto &item : out3) { ::cout << item << " ";
    }
}

TEST_CASE("Question #1.3") {
    execute_test("question_1_test_3.in", test_3);
}
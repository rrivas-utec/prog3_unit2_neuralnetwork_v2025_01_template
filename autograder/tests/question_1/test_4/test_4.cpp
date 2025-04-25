//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "neural_network.h"
using namespace std;

static void test_4_1() {
    Dense<float> d(2.0f, -1.0f);
    const std::vector<float> in = { 0.5f, -0.5f, 0.0f };
    const auto out = d.forward(in);
    // out[i] = in[i] * 2 + (-1)
    const std::vector<float> exp = { 0.0f, -2.0f, -1.0f };
    REQUIRE(out == exp);
}

static void test_4_2() {
    ReLU<int> r;
    std::vector<int> in = { -3, 0, 4, -1 };
    const auto out = r.forward(in);
    const std::vector<int> exp = { 0, 0, 4, 0 };
    REQUIRE(out == exp);
}


static void test_4_3() {
    Dropout<double> dr;
    const std::vector<double> in = { 0.1, 0.2, 0.3, 0.4, 0.5 };
    const auto out = dr.forward(in);
    // Debe tomar índices 1,3
    const std::vector<double> exp = { 0.2, 0.4 };
    REQUIRE(out == exp);

}


static void test_4_4() {
    Softmax<double> sm;
    const std::vector<double> in(4, 1.0); // todos iguales
    // Cada valor ≈ 0.25
    for (const auto out = sm.forward(in); auto v : out) {
        REQUIRE(v == Approx(0.25).epsilon(1e-6));
    }
}

static void test_4() {
    test_4_1();
    test_4_2();
    test_4_3();
    test_4_4();
}

TEST_CASE("Question #1.4") {
    execute_test("question_1_test_4.in", test_4);
}